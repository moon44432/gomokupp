import torch
from game import State
from mcts import pv_mcts_scores, ModelServer
from network import DN_OUTPUT_SIZE
from generate_data import first_player_value, write_data
from hparams import sp_temperature, sp_game_cnt, sp_num_cores
from tqdm import tqdm
import numpy as np
import multiprocessing


def play(model_server):
    """Play a single game using the shared model server"""
    history = []
    state = State()
    turn_cnt = 1

    while True:
        turn_cnt += 1

        if state.is_done():
            break

        scores = pv_mcts_scores(model_server, state, sp_temperature)

        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        action = np.random.choice(state.legal_actions(), p=scores)

        state = state.next(action)

    print(state)

    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def do_self_play(args):
    """Worker function for self-play with shared model server"""
    num, model_path = args
    
    # Create a model server for this worker
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_server = ModelServer(model_path, device=device, batch_size=8)
    
    history = []
    cnt = int(sp_game_cnt / sp_num_cores)
    
    for _ in tqdm(range(cnt), position=num, desc=f"Worker {num}"):
        h = play(model_server)
        history.extend(h)
    
    model_server.close()
    return history


def self_play():
    """Main self-play function using optimized parallelization"""
    model_path = './model/best.pth'
    
    # Prepare arguments for workers
    worker_args = [(i, model_path) for i in range(sp_num_cores)]
    
    # Use multiprocessing pool
    with multiprocessing.Pool(processes=sp_num_cores) as pool:
        history_list = pool.map(do_self_play, worker_args)
    
    # Combine all histories
    history = []
    for h in history_list:
        history += h

    write_data(history)
