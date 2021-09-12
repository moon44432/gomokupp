
from hparams import board_width, count_len

class State:
    board_width = board_width

    def __init__(self, pieces=None, enemy_pieces=None):
        self.pieces = pieces if pieces is not None else [0] * (board_width ** 2)
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [0] * (board_width ** 2)

    def count_piece(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for k in range(count_len):
                if y < 0 or board_width - 1 < y or x < 0 or board_width - 1 < x or \
                        self.enemy_pieces[x + y * board_width] == 0:
                    return False
                x, y = x + dx, y + dy
            return True

        for j in range(board_width):
            for i in range(board_width):
                if is_comp(i, j, 1, 0) or is_comp(i, j, 0, 1) or \
                        is_comp(i, j, 1, 1) or is_comp(i, j, -1, 1):
                    return True
        return False

    def is_draw(self):
        return self.count_piece(self.pieces) + self.count_piece(self.enemy_pieces) == board_width ** 2

    def is_done(self):
        return self.is_lose() or self.is_draw()

    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    def legal_actions(self):
        actions = []
        for i in range(self.board_width ** 2):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.count_piece(self.pieces) == self.count_piece(self.enemy_pieces)

    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(board_width ** 2):
            if i % board_width == 0:
                str += '{:2d} '.format(board_width - int(i / board_width))
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '.'
            if i % board_width == board_width - 1:
                str += '\n'
        str += '   ABCDEFGHIJKLMNO'
        return str
