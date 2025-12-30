from flask import Flask, render_template, jsonify, request, send_from_directory
import torch
from network import load_model
from game import State
from mcts import pv_mcts_action, ModelServer
from hparams import PLAY_MCTS_COUNT, PLAY_TEMPERATURE
from rule import renju
import threading

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
    
    session_id = str(len(game_sessions))
    
    with session_lock:
        game_sessions[session_id] = {
            'state': State(rule=renju),
            'player_color': player_color,
            'next_action': pv_mcts_action(model_server, PLAY_MCTS_COUNT, PLAY_TEMPERATURE)
        }
    
    # If player chose white, AI makes first move
    if player_color == 'white':
        with session_lock:
            session = game_sessions[session_id]
            action = session['next_action'](session['state'])
            session['state'] = session['state'].next(action)
            
            return jsonify({
                'session_id': session_id,
                'board': get_board_state(session['state']),
                'ai_move': action_to_coords(action, session['state'].board_width),
                'is_done': session['state'].is_done(),
                'winner': get_winner(session['state'])
            })
    
    return jsonify({
        'session_id': session_id,
        'board': get_board_state(game_sessions[session_id]['state']),
        'is_done': False,
        'winner': None
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
                'ai_move': None
            })
        
        # AI makes move
        action = session['next_action'](state)
        state = state.next(action)
        session['state'] = state
        
        return jsonify({
            'board': get_board_state(state),
            'ai_move': action_to_coords(action, state.board_width),
            'is_done': state.is_done(),
            'winner': get_winner(state)
        })


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """게임 리셋"""
    data = request.json
    session_id = data.get('session_id')
    player_color = data.get('player_color', 'black')
    
    if session_id not in game_sessions:
        return jsonify({'error': 'Invalid session'}), 400
    
    with session_lock:
        session = game_sessions[session_id]
        session['state'] = State()
        session['player_color'] = player_color
        
        # If player chose white, AI makes first move
        if player_color == 'white':
            action = session['next_action'](session['state'])
            session['state'] = session['state'].next(action)
            
            return jsonify({
                'board': get_board_state(session['state']),
                'ai_move': action_to_coords(action, session['state'].board_width),
                'is_done': session['state'].is_done(),
                'winner': get_winner(session['state'])
            })
    
    return jsonify({
        'board': get_board_state(game_sessions[session_id]['state']),
        'is_done': False,
        'winner': None
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
    
    return 'draw'


if __name__ == '__main__':
    initialize_model()
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
