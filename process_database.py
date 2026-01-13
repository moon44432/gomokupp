import xml.etree.ElementTree as etree
from tqdm import tqdm

from game import State, get_input_planes
from network import output_size
from generate_data import write_data
from hparams import RIF_DATABASE_PATH, XML_DATABASE_PATH, MAX_DATA_LENGTH, GAME_LENGTH_THRES, BOARD_WIDTH


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


def generate_records_from_rif():
    '''
    Example of game record in .rif file:
    <game id="171548" publisher="91" tournament="3369" round="8" rule="25" black="3069" white="3969" bresult="1" btime="8" wtime="8" opening="19" alt="51" swap="R--R-">
    <move>h8 h9 i8 g8 g9 f7 e6 f10 f6 g6 e8 e7 d7 c6 i10 i7 g7 i9 c8 f5 h5 j8 k9 c5 d5 d6 b4 f4 h6 k7 l6 h7 i5 j4 j5 k5 d4 e5 g3 k4 k6 m4 j6 i6 h4 f2 h3 h2 i4 l7 g2</move>
    <info>B=1245</info>
    </game>
    '''
    xmID = etree.parse(RIF_DATABASE_PATH)
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


def generate_records_from_xml():
    '''
    Example of game record in .xml file:
    <game>
    <id>0002</id>
    <creation_time>1062547200</creation_time>
    <black>warpod</black>
    <white>sai</white>
    <turn_color>black</turn_color>
    <board>h8 h9 h6 i10 i6 i9 g9 g8 j11 i7 i8 k7 g6 j6 f8 e6 f7 h10 j8 k10 j10 j9 k8 l8 i5 g7 h7 f5 d9 l9 k9 e8 f10 f9 d10 d8 h12 i11 d7 l10 l7 i13</board>
    <moves>h8 h9 h6 i10 i6 i9 g9 g8 j11 i7 i8 k7 g6 j6 f8 e6 f7 h10 j8 k10 j10 j9 k8 l8 i5 g7 h7 f5 d9 l9 k9 e8 f10 f9 d10 d8 h12 i11 d7 l10 l7 i13</moves>
    <alt5></alt5>
    <proposition></proposition>
    <proposer></proposer>
    <status>finished</status>
    <rule>classic</rule>
    <time>0</time>
    <tpm>0</tpm>
    <start_time>1062547200</start_time>
    <end_time>1062547200</end_time>
    <winner>white</winner>
    <winby>resign</winby>
    <time_left_black>0</time_left_black>
    <time_left_white>0</time_left_white>
    <tid></tid>
    </game>
    '''
    xmID = etree.parse(XML_DATABASE_PATH)
    root = xmID.getroot()
    cnt = 0
    record_list = []

    for game in tqdm(root):
        move_str = game.find("board").text
        winner = '1' if game.find("winner").text == 'black' else ('0' if game.find("winner").text == 'white' else '0.5')
        if move_str is None or '--' in move_str or len(move_str.split()) < GAME_LENGTH_THRES:
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
