from hparams import board_width

def _renju(state):
    directions = [(1, 0), (0, 1), (-1, 1), (1, 1)]
    open_threes = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    banned = [False] * (board_width ** 2)
    banned_overlines = [False] * (board_width ** 2)
    banned_44 = [False] * (board_width ** 2)
    banned_33 = [False] * (board_width ** 2)

    def is_legal(x, y):
        if x < 0 or x >= board_width or y < 0 or y >= board_width:
            return False
        elif state.pieces[x + y*board_width] == 1 or state.enemy_pieces[x + y*board_width] == 1 or banned[x + y*board_width]:
            return False
        return True
    
    def pattern_match(pos, dir, pattern):
        x, y = divmod(pos, board_width)
        dx, dy = dir
        is_match = True
        open_cnt = 0
        for i in range(len(pattern)):
            x_i = x - dx * (i + 1)
            y_i = y - dy * (i + 1)
            is_match = True
            open_cnt = 0
            if is_legal(x_i, y_i):
                open_cnt += 1
            for p in pattern:
                x_i, y_i = x_i + dx, y_i + dy
                if 0 <= x_i < board_width and 0 <= y_i < board_width:
                    if p == 1:
                        if state.pieces[x_i + y_i*board_width] != 1:
                            is_match = False
                            break
                    elif p == 0:
                        if not is_legal(x_i, y_i):
                            is_match = False
                            break
                else:
                    is_match = False
                    break
            if is_legal(x_i + dx, y_i + dy):
                open_cnt += 1
            if is_match:
                return True, open_cnt
        return False, open_cnt

    def count_consecutive(pos, dir):
        x, y = divmod(pos, board_width)
        dx, dy = dir
        count = 1
        open_cnt = 0
        while True:
            x, y = x + dx, y + dy
            if 0 <= x < board_width and 0 <= y < board_width:
                if state.pieces[x + y*board_width] == 1:
                    count += 1
                else: break
            else: break
        if is_legal(x, y):
            open_cnt += 1
        x, y = divmod(pos, board_width)
        while True:
            x, y = x - dx, y - dy
            if 0 <= x < board_width and 0 <= y < board_width:
                if state.pieces[x + y*board_width] == 1:
                    count += 1
                else: break
            else: break
        if is_legal(x, y):
            open_cnt += 1
        return count, open_cnt
    
    def chk_overline(pos, dir):
        count, _ = count_consecutive(pos, dir)
        if count >= 6:
            return True
        return False

    def chk_five(pos, dir):
        count, _ = count_consecutive(pos, dir)
        if count == 5:
            return True
        return False
    
    def chk_four(pos, dir):
        count, open_cnt = count_consecutive(pos, dir)
        if count == 4 and open_cnt >= 1:
            return True
        return False
    
    def chk_open_four(pos, dir):
        count, open_cnt = count_consecutive(pos, dir)
        if count == 4 and open_cnt == 2:
            return True
        return False
    
    def chk_open_three(pos, dir):
        for pattern in open_threes:
            is_match, open_cnt = pattern_match(pos, dir, pattern)
            if is_match and open_cnt == 2:
                return True
        return False
    
    def ban_overline():
        for pos in range(board_width ** 2):
            state.pieces[pos] = 1
            for dir in directions:
                if chk_overline(pos, dir):
                    banned_overlines[pos] = True
                    break
            state.pieces[pos] = 0
    
    def ban_44():
        for pos in range(board_width ** 2):
            state.pieces[pos] = 1
            four_cnt = 0
            for dir in directions:
                if chk_four(pos, dir):
                    four_cnt += 1
            if four_cnt >= 2:
                banned_44[pos] = True
            state.pieces[pos] = 0

    def ban_33():
        for pos in range(board_width ** 2):
            state.pieces[pos] = 1
            three_cnt = 0
            for dir in directions + [(-dir[0], -dir[1]) for dir in directions]:
                if chk_open_three(pos, dir):
                    three_cnt += 1
            if three_cnt >= 2:
                banned_33[pos] = True
            state.pieces[pos] = 0

    def allow_five():
        for pos in range(board_width ** 2):
            state.pieces[pos] = 1
            for dir in directions:
                if chk_five(pos, dir):
                    banned[pos] = False
                    break
            state.pieces[pos] = 0

    if state.is_first_player():
        ban_overline()
        banned = [banned[i] or banned_overlines[i] for i in range(board_width ** 2)]
        ban_44()
        banned = [banned[i] or banned_44[i] for i in range(board_width ** 2)]
        ban_33()
        banned = [banned[i] or banned_33[i] for i in range(board_width ** 2)]
        allow_five()

    return banned

def renju(state):
    banned = _renju(state)
    actions = []
    for i in range(board_width ** 2):
        if state.pieces[i] == 0 and state.enemy_pieces[i] == 0 and not banned[i]:
            actions.append(i)

    return actions