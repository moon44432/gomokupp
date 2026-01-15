import torch
from network import load_model
from game import State
from mcts import pv_mcts_action, ModelServer
from hparams import PLAY_TEMPERATURE, PLAY_MCTS_COUNT
from rule import Renju


class Game:
    def __init__(self, rule=None, model_server=None):
        self.state = State(rule=rule)
        self.next_action = pv_mcts_action(model_server, PLAY_MCTS_COUNT, PLAY_TEMPERATURE)

    def turn_of_human(self):
        x, y = input('A~O 1~15 형태로 입력: ').split()

        x = ord(x.upper()) - ord('A')
        y = self.state.board_width - int(y)

        if x < 0 or self.state.board_width - 1 < x or y < 0 or self.state.board_width - 1 < y:
            return
        action = x + y * self.state.board_width

        if not (action in self.state.legal_actions()):
            return

        self.state = self.state.next(action)
        self.draw()

    def turn_of_ai(self):
        action = self.next_action(self.state)

        self.state = self.state.next(action)
        self.draw()

    def draw(self):
        print(self.state)


if __name__ == '__main__':
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Create model server for efficient inference
    model_server = ModelServer('./model/best.pth', device=device)

    # 게임 UI 실행
    f = Game(model_server=model_server, rule=Renju())

    while True:
        f.state = State(rule=Renju())
        while True:
            # f.turn_of_ai()
            f.turn_of_human()
            if f.state.is_done():
                print('대국 종료')
                break

            f.turn_of_human()
            if f.state.is_done():
                print('대국 종료')
                break
