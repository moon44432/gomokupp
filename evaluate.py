import torch
import multiprocessing
from shutil import copy
from tqdm import tqdm

from hparams import EN_GAME_COUNT, EN_TEMPERATURE, EN_AVERAGE_POINT, EN_NUM_CORES, EN_MCTS_COUNT
from game import State
from mcts import pv_mcts_action, ModelServer


def first_player_point(end_state):
    if end_state.is_lose():
        return 0 if end_state.is_first_player() else 1
    if end_state.is_forbidden_move():
        return 0 if not end_state.is_first_player() else 1
    return 0.5


def play(next_actions, rule=None):
    state = State(rule=rule)

    while True:
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)


def update_best_player():
    copy('./model/latest.pth', './model/best.pth')
    print('Updating best player...')


def do_evaluate(args):
    """Evaluate using shared model servers"""
    num, latest_path, best_path, rule = args
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model servers for batch inference
    model_server0 = ModelServer(latest_path, device=device)
    model_server1 = ModelServer(best_path, device=device)

    next_action0 = pv_mcts_action(model_server0, EN_MCTS_COUNT, EN_TEMPERATURE)
    next_action1 = pv_mcts_action(model_server1, EN_MCTS_COUNT, EN_TEMPERATURE)
    next_actions = (next_action0, next_action1)
    total_point = 0

    cnt = int(EN_GAME_COUNT / EN_NUM_CORES)
    for i in tqdm(range(cnt), position=num, desc=f"Eval {num}"):
        if i % 2 == 0:
            total_point += play(next_actions, rule)
        else:
            total_point += 1 - play(list(reversed(next_actions)), rule)

    model_server0.close()
    model_server1.close()

    return total_point


def evaluate_network(rule=None):
    latest_path = './model/latest.pth'
    best_path = './model/best.pth'
    
    # Prepare arguments for workers
    worker_args = [(i, latest_path, best_path, rule) for i in range(EN_NUM_CORES)]
    
    with multiprocessing.Pool(processes=EN_NUM_CORES) as pool:
        total = pool.map(do_evaluate, worker_args)
    
    total_point = sum(total)
    print(total, total_point)

    average_point = total_point / (int(EN_GAME_COUNT / EN_NUM_CORES) * EN_NUM_CORES)
    print('Average point: ', average_point)

    if average_point > EN_AVERAGE_POINT:
        update_best_player()
        return True
    else:
        return False
