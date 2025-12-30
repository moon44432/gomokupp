import torch
import torch.multiprocessing as mp
from math import sqrt
from network import DN_INPUT_SHAPE
import numpy as np
import queue
import threading


class ModelServer:
    """Shared model server for efficient batch prediction"""
    def __init__(self, model_path, device='cpu', batch_size=8):
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
        c, a, b = DN_INPUT_SHAPE
        batch = []
        for state in states:
            x = np.array([state.pieces, state.enemy_pieces])
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
        c, a, b = DN_INPUT_SHAPE
        x = np.array([state.pieces, state.enemy_pieces])
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


def pv_mcts_scores(model, state, temperature, eval_cnt):
    class Node:
        def __init__(self, state, p):
            self.state = state
            self.p = p
            self.w = 0
            self.n = 0
            self.child_nodes = None

        def evaluate(self):
            if self.state.is_done():
                value = -1 if self.state.is_lose() else 0

                self.w += value
                self.n += 1
                return value

            if not self.child_nodes:
                policies, value = predict(model, self.state)

                self.w += value
                self.n += 1

                self.child_nodes = []
                for action, policy in zip(self.state.legal_actions(), policies):
                    self.child_nodes.append(Node(self.state.next(action), policy))
                return value

            else:
                value = -self.next_child_node().evaluate()

                self.w += value
                self.n += 1
                return value

        def next_child_node(self):
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            return self.child_nodes[np.argmax(pucb_values)]

    root_node = Node(state, 0)

    for _ in range(eval_cnt):
        root_node.evaluate()

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
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]
