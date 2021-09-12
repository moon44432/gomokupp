
from tqdm import tqdm
from game import State
from network import DN_OUTPUT_SIZE
from generate_data import first_player_value, write_data
from hparams import xml_path, max_record_cnt, board_width
import xml.etree.ElementTree as etree


def play(record):
    history = []
    actions = [int(i) for i in record.split()]
    state = State()

    for action in actions:
        policies = [0] * DN_OUTPUT_SIZE
        policies[action] = 1

        history.append([[state.pieces, state.enemy_pieces], policies, None])
        state = state.next(action)

    value = first_player_value(state)

    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def record_play(record):
    history = []

    for i in range(len(record)):
        h = play(record[i])
        history.extend(h)

    write_data(history)


def get_record_str(record_str):
    result_str = ''
    history = record_str.split()

    for i in range(len(history)):
        result_str = result_str + \
                     str(board_width * (board_width - int(history[i][1:])) + ord(history[i][0]) - ord('a')) \
                         + ' '
    result_str += '\n'
    return result_str


def generate_record_list():
    xmID = etree.parse(xml_path)
    root = xmID.getroot()
    cnt = 0
    record_list = []

    for game in tqdm(root):
        if game.find("winby").text == "five":
            record = game.find("board").text
            if '--' in record:
                continue
            else:
                record_list.append(get_record_str(record))
                cnt += 1
        if cnt == max_record_cnt:
            break

    return record_list
