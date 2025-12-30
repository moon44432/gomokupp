
from datetime import datetime
import pickle
import os


def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0


def write_data(history, typ):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}_{}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second, typ)

    with open(path, mode='wb') as f:
        pickle.dump(history, f)
