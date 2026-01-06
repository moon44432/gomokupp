import torch
import numpy as np
import multiprocessing
from tqdm import tqdm

from game import State, get_input_planes
from mcts import pv_mcts_scores, ModelServer
from network import output_size
from generate_data import first_player_value, write_data
from hparams import RL_TEMP, RL_GAME_CNT, RL_CORES, RL_MCTS_CNT


def play(model_server, rule):
    """Play a single game using the shared model server"""
    history = []
    state = State(rule=rule)

    while True:
        if state.is_done():
            break

        scores = pv_mcts_scores(model_server, RL_MCTS_CNT, state, RL_TEMP)

        policies = [0] * output_size
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([get_input_planes(state), policies, None])

        action = np.random.choice(state.legal_actions(), p=scores)
        state = state.next(action)

    # print(state)

    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def do_self_play(args):
    """Worker function for self-play with shared model server"""
    num, model_path, rule = args
    
    # Create a model server for this worker
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_server = ModelServer(model_path, device=device, batch_size=8)
    
    history = []
    cnt = int(RL_GAME_CNT / RL_CORES)
    
    for _ in tqdm(range(cnt), position=num, desc=f"Worker {num}"):
        h = play(model_server, rule)
        history.extend(h)
    
    model_server.close()
    return history


def self_play(rule=None):
    """Main self-play function using optimized parallelization"""
    model_path = './model/best.pth'
    
    # Prepare arguments for workers
    worker_args = [(i, model_path, rule) for i in range(RL_CORES)]
    
    # Use multiprocessing pool
    with multiprocessing.Pool(processes=RL_CORES) as pool:
        history_list = pool.map(do_self_play, worker_args)
    
    # Combine all histories
    history = []
    for h in history_list:
        history += h

    write_data(history, 'sp')
