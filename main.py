
from tensorflow.keras.models import load_model
from game import State
from mcts import pv_mcts_action
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)


class Game:
    def __init__(self, model=None):
        self.state = State()
        self.next_action = pv_mcts_action(model, 0.1)

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
    # 베스트 플레이어 모델 로드
    model = load_model('./model/best.h5')

    # 게임 UI 실행
    f = Game(model=model)

    while True:
        f.state = State()
        while True:
            f.turn_of_ai()
            if f.state.is_done():
                print('대국 종료')
                break

            f.turn_of_human()
            if f.state.is_done():
                print('대국 종료')
                break
