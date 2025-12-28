
from keras.models import load_model
from keras import backend as K
from game import State
from mcts import pv_mcts_scores
from network import DN_OUTPUT_SIZE
from generate_data import first_player_value, write_data
from hparams import sp_temperature, sp_game_cnt, sp_num_cores
from tqdm import tqdm
import numpy as np
import multiprocessing


def play(model):
    history = []
    state = State()
    turn_cnt = 1

    while True:
        turn_cnt += 1

        if state.is_done():
            break

        scores = pv_mcts_scores(model, state, sp_temperature)

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


def do_self_play(num):
    history = []
    model = load_model('./model/best.h5')

    cnt = int(sp_game_cnt / sp_num_cores)
    for _ in tqdm(range(cnt)):
        h = play(model)
        history.extend(h)

    K.clear_session()
    del model
    return history


def self_play():
    pool = multiprocessing.Pool(processes=sp_num_cores)
    history_list = pool.map(do_self_play, range(sp_num_cores))
    pool.close()
    pool.join()

    history = []

    for h in history_list:
        history += h

    write_data(history)
