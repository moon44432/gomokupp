import torch
import threading
from flask import Flask, jsonify, request, send_from_directory

from game import State
from mcts import pv_mcts_action, ModelServer
from hparams import PLAY_MCTS_COUNT, PLAY_TEMPERATURE
from rule import Renju

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Global model server
model_server = None
device = None

# Game sessions (simple in-memory storage)
game_sessions = {}
session_lock = threading.Lock()


def initialize_model():
    global model_server, device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    model_server = ModelServer('./model/best.pth', device=device, batch_size=8)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """새 게임 시작"""
    data = request.json
    player_color = data.get('player_color', 'black')  # 'black' or 'white'
    use_renju = data.get('use_renju', True)  # 렌주룰 사용 여부
    use_ai = data.get('use_ai', True)  # AI 사용 여부
    
    session_id = str(len(game_sessions))
    
    rule = Renju() if use_renju else None
    
    with session_lock:
        game_sessions[session_id] = {
            'state': State(rule=rule),
            'player_color': player_color,
            'use_ai': use_ai,
            'use_renju': use_renju,
            'next_action': pv_mcts_action(model_server, PLAY_MCTS_COUNT, PLAY_TEMPERATURE) if use_ai else None
        }
    
    # If AI is enabled and player chose white, AI makes first move
    if use_ai and player_color == 'white':
        with session_lock:
            session = game_sessions[session_id]
            action = session['next_action'](session['state'])
            session['state'] = session['state'].next(action)
            
            return jsonify({
                'session_id': session_id,
                'board': get_board_state(session['state']),
                'ai_move': action_to_coords(action, session['state'].board_width),
                'is_done': session['state'].is_done(),
                'winner': get_winner(session['state']),
                'forbidden_moves': get_forbidden_moves(session['state']) if use_renju else []
            })
    
    return jsonify({
        'session_id': session_id,
        'board': get_board_state(game_sessions[session_id]['state']),
        'is_done': False,
        'winner': None,
        'forbidden_moves': get_forbidden_moves(game_sessions[session_id]['state']) if use_renju else []
    })


@app.route('/api/move', methods=['POST'])
def make_move():
    """플레이어가 수를 둠"""
    data = request.json
    session_id = data.get('session_id')
    x = data.get('x')
    y = data.get('y')
    
    if session_id not in game_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    with session_lock:
        session = game_sessions[session_id]
        state = session['state']
        use_ai = session.get('use_ai', True)
        use_renju = session.get('use_renju', True)
        
        # Convert coordinates to action
        action = x + y * state.board_width
        
        # Check if move is legal
        if action not in state.legal_actions():
            return jsonify({'error': 'Illegal move'}), 400
        
        # Make player move
        state = state.next(action)
        session['state'] = state
        
        # Check if game is done
        if state.is_done():
            return jsonify({
                'board': get_board_state(state),
                'is_done': True,
                'winner': get_winner(state),
                'ai_move': None,
                'forbidden_moves': get_forbidden_moves(state) if use_renju else []
            })
        
        # If AI is enabled, AI makes move
        if use_ai:
            action = session['next_action'](state)
            state = state.next(action)
            session['state'] = state
            
            return jsonify({
                'board': get_board_state(state),
                'ai_move': action_to_coords(action, state.board_width),
                'is_done': state.is_done(),
                'winner': get_winner(state),
                'forbidden_moves': get_forbidden_moves(state) if use_renju else []
            })
        else:
            # No AI, just return current state
            return jsonify({
                'board': get_board_state(state),
                'ai_move': None,
                'is_done': state.is_done(),
                'winner': get_winner(state),
                'forbidden_moves': get_forbidden_moves(state) if use_renju else []
            })


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """게임 리셋"""
    data = request.json
    session_id = data.get('session_id')
    player_color = data.get('player_color', 'black')
    use_renju = data.get('use_renju', True)
    use_ai = data.get('use_ai', True)
    
    if session_id not in game_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    rule = Renju() if use_renju else None
    
    with session_lock:
        session = game_sessions[session_id]
        session['state'] = State(rule=rule)
        session['player_color'] = player_color
        session['use_ai'] = use_ai
        session['use_renju'] = use_renju
        session['next_action'] = pv_mcts_action(model_server, PLAY_MCTS_COUNT, PLAY_TEMPERATURE) if use_ai else None
        
        # If AI is enabled and player chose white, AI makes first move
        if use_ai and player_color == 'white':
            action = session['next_action'](session['state'])
            session['state'] = session['state'].next(action)
            
            return jsonify({
                'board': get_board_state(session['state']),
                'ai_move': action_to_coords(action, session['state'].board_width),
                'is_done': session['state'].is_done(),
                'winner': get_winner(session['state']),
                'forbidden_moves': get_forbidden_moves(session['state']) if use_renju else []
            })
    
    return jsonify({
        'board': get_board_state(game_sessions[session_id]['state']),
        'is_done': False,
        'winner': None,
        'forbidden_moves': get_forbidden_moves(game_sessions[session_id]['state']) if use_renju else []
    })


def get_board_state(state):
    """보드 상태를 배열로 변환 (0: 빈칸, 1: 흑돌, 2: 백돌)"""
    board = []
    board_size = state.board_width * state.board_width
    
    for i in range(board_size):
        if state.pieces[i] == 1:
            # Current player's piece
            board.append(int(1 if state.is_first_player() else 2))
        elif state.enemy_pieces[i] == 1:
            # Opponent's piece
            board.append(int(2 if state.is_first_player() else 1))
        else:
            board.append(0)
    
    return board


def action_to_coords(action, board_width):
    """action을 (x, y) 좌표로 변환"""
    x = int(action % board_width)
    y = int(action // board_width)
    return {'x': x, 'y': y}


def get_winner(state):
    """승자 반환 (None, 'black', 'white', 'draw')"""
    if not state.is_done():
        return None
    
    if state.is_lose():
        # Current player lost, so the opponent won
        # Need to determine who is who
        return 'white' if state.is_first_player() else 'black'
    
    if state.is_forbidden_move():
        # Current player made forbidden move, so opponent won
        return 'white' if not state.is_first_player() else 'black'
    
    return 'draw'


def get_forbidden_moves(state):
    """렌주룰 금수 위치 반환"""
    if state.rule is None:
        return []
    banned = state.rule.get_banned(state)
    forbidden = []
    for i in range(len(banned)):
        if banned[i]:
            x = i % state.board_width
            y = i // state.board_width
            # 금수 타입: 1=장목(6+), 2=44, 3=33
            type_map = {1: '6+', 2: '44', 3: '33'}
            forbidden.append({'x': x, 'y': y, 'type': type_map.get(banned[i], '')})
    return forbidden


if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
