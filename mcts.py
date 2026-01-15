import torch
import numpy as np
import queue
import threading
import concurrent.futures
from math import sqrt

from game import get_input_planes
from network import input_shape
from hparams import MCTS_VIRTUAL_LOSS, MCTS_NUM_THREADS


class ModelServer:
    """Shared model server for efficient batch prediction"""
    def __init__(self, model_path, device='cpu', batch_size=16):
        from network import load_model
        self.model = load_model(model_path)
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        
        # Queue for prediction requests
        self.request_queue = queue.Queue()
        self.shutdown = False
        
        # Start prediction thread
        self.prediction_thread = threading.Thread(target=self._prediction_worker, daemon=True)
        self.prediction_thread.start()
    
    def _prediction_worker(self):
        """Worker thread that processes prediction requests in batches"""
        while not self.shutdown:
            batch_requests = []
            
            # Collect requests up to batch_size with timeout
            try:
                # Get first request (blocking)
                first_request = self.request_queue.get(timeout=0.1)
                batch_requests.append(first_request)
                
                # Try to get more requests to fill batch (non-blocking)
                while len(batch_requests) < self.batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        batch_requests.append(request)
                    except queue.Empty:
                        break
                
                if not batch_requests:
                    continue
                
                # Extract states and result queues
                states = [req[0] for req in batch_requests]
                result_queues = [req[1] for req in batch_requests]
                
                # Batch prediction
                with torch.no_grad():
                    batch_x = self._prepare_batch(states)
                    batch_x = batch_x.to(self.device)
                    policies, values = self.model(batch_x)
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()
                
                # Send results back
                for i, result_queue in enumerate(result_queues):
                    result_queue.put((policies[i], values[i][0]))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in prediction worker: {e}")
    
    def _prepare_batch(self, states):
        """Prepare batch of states for model input"""
        c, a, b = input_shape
        batch = []
        for state in states:
            x = np.array(get_input_planes(state))
            x = x.reshape(c, a, b)
            batch.append(x)
        return torch.FloatTensor(np.array(batch))
    
    def predict(self, state):
        """Request prediction for a single state"""
        result_queue = queue.Queue()
        self.request_queue.put((state, result_queue))
        policies, value = result_queue.get()
        
        # Filter to legal actions
        legal_policies = policies[list(state.legal_actions())]
        legal_policies /= legal_policies.sum() if legal_policies.sum() > 0 else 1
        
        return legal_policies, value
    
    def close(self):
        """Shutdown the model server"""
        self.shutdown = True
        self.prediction_thread.join()


def predict(model, state):
    """Prediction function compatible with both ModelServer and direct model"""
    if isinstance(model, ModelServer):
        return model.predict(state)
    else:
        # Direct model inference (legacy support)
        c, a, b = input_shape
        x = np.array(get_input_planes(state))
        x = x.reshape(1, c, a, b)
        x = torch.FloatTensor(x)
        
        with torch.no_grad():
            model.eval()
            policies, value = model(x)
            policies = policies[0].numpy()
            value = value[0][0].item()
        
        legal_policies = policies[list(state.legal_actions())]
        legal_policies /= legal_policies.sum() if legal_policies.sum() > 0 else 1
        
        return legal_policies, value


def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

class Node:
    def __init__(self, state, p):
        self.state = state
        self.p = p
        self.w = 0.0
        self.n = 0
        self.child_nodes = None
        self.lock = threading.Lock()

    def evaluate(self, model):
        if self.state.is_done():
            if self.state.is_lose():
                value = -1
            elif self.state.is_forbidden_move():
                value = 1
            else:
                value = 0
            
            with self.lock:
                self.w += value
                self.n += 1
            return value

        # Check for children with lock
        with self.lock:
            has_children = self.child_nodes is not None

        if not has_children:
            policies, value = predict(model, self.state)

            with self.lock:
                self.w += value
                self.n += 1
                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
            return value

        else:
            # Standard PUCT selection
            C_PUCT = 1.5
            with self.lock:
                # Calculate scores for all children
                t = sum(c.n for c in self.child_nodes)
                # Optimization: pre-calculate sqrt(t)
                sqrt_t = sqrt(t)
                
                best_value = -float('inf')
                best_child = None
                
                for child in self.child_nodes:
                    q = -child.w / child.n if child.n else 0.0
                    u = C_PUCT * child.p * sqrt_t / (1 + child.n)
                    score = q + u
                    
                    if score > best_value:
                        best_value = score
                        best_child = child
                
                # Apply Virtual Loss
                with best_child.lock:
                    best_child.n += MCTS_VIRTUAL_LOSS
                    best_child.w += MCTS_VIRTUAL_LOSS

            # Evaluate child (Recursive)
            value = -best_child.evaluate(model)

            # Remove Virtual Loss
            with best_child.lock:
                best_child.n -= MCTS_VIRTUAL_LOSS
                best_child.w -= MCTS_VIRTUAL_LOSS

            # Update self
            with self.lock:
                self.w += value
                self.n += 1
                
            return value

def pv_mcts_scores(model, state, temperature, eval_cnt):
    root_node = Node(state, 0)
    
    # Use thread pool to parallelize MCTS
    num_threads = MCTS_NUM_THREADS
    iters_per_thread = (eval_cnt + num_threads - 1) // num_threads

    def worker(iterations):
        for _ in range(iterations):
            root_node.evaluate(model)

    # If eval_cnt is small, or just 1 thread, don't submit overhead
    if num_threads <= 1 or eval_cnt < num_threads:
        worker(eval_cnt)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, iters_per_thread) for _ in range(num_threads)]
            concurrent.futures.wait(futures)

    scores = nodes_to_scores(root_node.child_nodes)

    if temperature == 0:
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:
        scores = boltzman(scores, temperature)
    return scores


def pv_mcts_action(model, eval_count, temperature=0):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, eval_count)
        return np.random.choice(state.legal_actions(), p=scores)

    return pv_mcts_action


def boltzman(xs, temperature):
    # AlphaGo Zero-style sampling: p(a) ∝ N(a)^(1/τ)
    # Compute in log-space to avoid overflow when τ is small.
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return []

    if temperature <= 0:
        probs = np.zeros_like(xs)
        probs[int(np.argmax(xs))] = 1.0
        return probs.tolist()

    # Keep exact behavior for zero-count moves: 0^(1/τ) = 0
    log_xs = np.where(xs > 0, np.log(xs), -np.inf)
    finite = np.isfinite(log_xs)

    # If all visit counts are zero, fall back to uniform
    if not np.any(finite):
        return (np.ones_like(xs) / xs.size).tolist()

    logits = log_xs / float(temperature)
    max_logit = np.max(logits[finite])
    logits = logits - max_logit

    exp_logits = np.zeros_like(xs)
    exp_logits[finite] = np.exp(logits[finite])
    s = exp_logits.sum()
    if not np.isfinite(s) or s <= 0:
        # Fallback: put all mass on argmax visit count
        probs = np.zeros_like(xs)
        probs[int(np.argmax(xs))] = 1.0
        return probs.tolist()

    return (exp_logits / s).tolist()
