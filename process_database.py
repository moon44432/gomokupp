import xml.etree.ElementTree as etree
from tqdm import tqdm

from game import State, get_input_planes
from network import output_size
from generate_data import write_data
from hparams import DATABASE_PATH, MAX_DATA_LENGTH, GAME_LENGTH_THRES, BOARD_WIDTH


def play(record):
    history = []
    moves, winner = record
    state = State()

    for action in moves:
        policies = [0] * output_size
        policies[action] = 1

        history.append([get_input_planes(state), policies, None])
        state = state.next(action)

    if winner == '1':
        value = 1
    elif winner == '0':
        value = -1
    else:
        value = 0

    for i in range(len(history)):
        history[i][2] = value
        value = -value

    return history


def generate_records():
    '''
    Example of game record in .rif file:
    <game id="171548" publisher="91" tournament="3369" round="8" rule="25" black="3069" white="3969" bresult="1" btime="8" wtime="8" opening="19" alt="51" swap="R--R-">
    <move>h8 h9 i8 g8 g9 f7 e6 f10 f6 g6 e8 e7 d7 c6 i10 i7 g7 i9 c8 f5 h5 j8 k9 c5 d5 d6 b4 f4 h6 k7 l6 h7 i5 j4 j5 k5 d4 e5 g3 k4 k6 m4 j6 i6 h4 f2 h3 h2 i4 l7 g2</move>
    <info>B=1245</info>
    </game>
    '''
    xmID = etree.parse(DATABASE_PATH)
    root = xmID.find("games")
    cnt = 0
    record_list = []

    for game in tqdm(root):
        move_str = game.find("move").text
        # bresult: '1'(black win), '0'(white win), '0.5'(draw)
        winner = game.get("bresult")
        if move_str == None or len(move_str.split()) < GAME_LENGTH_THRES:
            continue
        else:
            moves = move_str.split()
            record_list.append(
                ([BOARD_WIDTH * (BOARD_WIDTH - int(move[1:])) + ord(move[0]) - ord('a') for move in moves]
                , winner)
            )
            cnt += 1

        if cnt == MAX_DATA_LENGTH:
            break

    return record_list
