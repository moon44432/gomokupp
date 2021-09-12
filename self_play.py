
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from game import State
from mcts import pv_mcts_scores
from network import DN_OUTPUT_SIZE
from generate_data import first_player_value, write_data
from hparams import sp_temperature, sp_game_cnt
from tqdm import tqdm
import numpy as np


def play(model):
    history = []
    state = State()
    turn_cnt = 1

    while True:
        print('  Turn {:d}'.format(turn_cnt))
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


def self_play():
    history = []
    model = load_model('./model/best.h5')

    for i in tqdm(range(sp_game_cnt)):
        print('\rSelf play {}/{}'.format(i + 1, sp_game_cnt), end='')

        h = play(model)
        history.extend(h)

    print('')
    write_data(history)

    K.clear_session()
    del model
