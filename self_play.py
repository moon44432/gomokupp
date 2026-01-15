import torch
import numpy as np
import threading
import concurrent.futures
from tqdm import tqdm

from game import State, get_input_planes
from mcts import pv_mcts_scores, ModelServer
from network import output_size
from generate_data import first_player_value, write_data
from hparams import RL_TEMP, RL_GAME_COUNT, RL_CORES, RL_MCTS_COUNT


def play(model_server, rule):
    """Play a single game using the shared model server"""
    history = []
    state = State(rule=rule)

    while True:
        if state.is_done():
            break

        scores = pv_mcts_scores(model_server, state, RL_TEMP, RL_MCTS_COUNT)

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
    """Worker function for self-play"""
    model_server, rule, cnt, pbar = args
    history = []
    
    for _ in range(cnt):
        h = play(model_server, rule)
        history.extend(h)
        # Update progress bar safely
        with pbar.get_lock():
            pbar.update(1)
            
    return history


def self_play(rule=None):
    """Main self-play function using threading and shared ModelServer"""
    model_path = './model/best.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize shared model server
    # Adjust batch_size as needed. 16 ~ 32 is usually good.
    model_server = ModelServer(model_path, device=device)
    
    total_games = RL_GAME_COUNT
    num_threads = RL_CORES
    games_per_thread = total_games // num_threads
    
    # Use a shared tqdm progress bar
    with tqdm(total=total_games, desc="Self Play") as pbar:
        # Prepare arguments: (server, rule, count, pbar)
        worker_args = [(model_server, rule, games_per_thread, pbar) for _ in range(num_threads)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            game_histories = list(executor.map(do_self_play, worker_args))
    
    model_server.close()
    
    # Combine all histories
    history = []
    for h in game_histories:
        history += h

    write_data(history, 'sp')
