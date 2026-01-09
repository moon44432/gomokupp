from hparams import BOARD_WIDTH, COUNT_LEN, PREV_STATE_COUNT

class State:
    def __init__(self, pieces=None, enemy_pieces=None, rule=None, last_move=None, prev_state=None):
        self.board_width = BOARD_WIDTH
        self.pieces = pieces if pieces is not None else [0] * (BOARD_WIDTH ** 2)
        self.enemy_pieces = enemy_pieces if enemy_pieces is not None else [0] * (BOARD_WIDTH ** 2)
        self.rule = rule
        self.last_move = last_move
        self.prev_state = prev_state

    def count_piece(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count

    def is_lose(self):
        def is_comp(x, y, dx, dy):
            for _ in range(COUNT_LEN):
                if y < 0 or BOARD_WIDTH - 1 < y or x < 0 or BOARD_WIDTH - 1 < x or \
                        self.enemy_pieces[x + y * BOARD_WIDTH] == 0:
                    return False
                x, y = x + dx, y + dy
            return True

        for j in range(BOARD_WIDTH):
            for i in range(BOARD_WIDTH):
                if is_comp(i, j, 1, 0) or is_comp(i, j, 0, 1) or \
                        is_comp(i, j, 1, 1) or is_comp(i, j, -1, 1):
                    return True
        return False

    def is_draw(self):
        return self.count_piece(self.pieces) + self.count_piece(self.enemy_pieces) == BOARD_WIDTH ** 2

    def is_forbidden_move(self):
        if self.last_move is None or self.rule is None:
            return False
        
        if self.is_first_player():
            return False
            
        self.enemy_pieces[self.last_move] = 0
        is_banned = self.rule.is_banned(self.enemy_pieces, self.pieces, self.last_move)
        self.enemy_pieces[self.last_move] = 1
        
        return is_banned != 0

    def is_done(self):
        return self.is_lose() or self.is_draw() or self.is_forbidden_move()

    def next(self, action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces, rule=self.rule, last_move=action, prev_state=self)

    def legal_actions(self):
        if self.rule is not None:
            return self.rule.legal_actions(self)
        actions = []
        for i in range(BOARD_WIDTH ** 2):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions

    def is_first_player(self):
        return self.count_piece(self.pieces) == self.count_piece(self.enemy_pieces)

    def __str__(self):
        ox = ('o ', 'x ') if self.is_first_player() else ('x ', 'o ')
        str = ''
        banned = self.rule.get_banned(self) if self.rule is not None else [False] * (BOARD_WIDTH ** 2)
        for i in range(BOARD_WIDTH ** 2):
            if i % BOARD_WIDTH == 0:
                str += '{:2d} '.format(BOARD_WIDTH - int(i / BOARD_WIDTH))
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                if banned[i]:
                    str += '_ '
                else:
                    str += '. '
            if i % BOARD_WIDTH == BOARD_WIDTH - 1:
                str += '\n'
        str += '   A B C D E F G H I J K L M N O'
        return str


def get_input_planes(state):
    planes = []
    
    current_state = state
    for i in range(PREV_STATE_COUNT + 1):
        if current_state is None:
            planes.append([0] * (state.board_width ** 2)) # My stones
        else:
            if i % 2 == 0:
                planes.append(current_state.pieces)
            else:
                planes.append(current_state.enemy_pieces)
            
            current_state = current_state.prev_state

    current_state = state
    for i in range(PREV_STATE_COUNT + 1):
        if current_state is None:
            planes.append([0] * (state.board_width ** 2)) # Opponent stones
        else:
            if i % 2 == 0:
                planes.append(current_state.enemy_pieces)
            else:
                planes.append(current_state.pieces)
            
            current_state = current_state.prev_state

    # Color plane
    color = 1 if state.is_first_player() else 0
    planes.append([color] * (state.board_width ** 2))
    
    return planes
