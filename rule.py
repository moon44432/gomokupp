from hparams import BOARD_WIDTH

class Rule:
    def legal_actions(self, state):
        actions = []
        for i in range(BOARD_WIDTH ** 2):
            if state.pieces[i] == 0 and state.enemy_pieces[i] == 0:
                actions.append(i)
        return actions
    
    def is_banned(self, pieces, enemy_pieces, pos):
        return 0
    
class Renju(Rule):
    board_width = BOARD_WIDTH
    directions = [(1, 0), (0, 1), (-1, 1), (1, 1)]

    fours = [[-1, 0, 1, 1, 1, 1, -1],
             [-1, 1, 1, 1, 1, 0, -1],
             [-1, 1, 0, 1, 1, 1, -1],
             [-1, 1, 1, 0, 1, 1, -1],
             [-1, 1, 1, 1, 0, 1, -1]]
    
    open_three_patterns = [
            ([-1, 0, 0, 1, 1, 1, 0, -1], 2),
            ([-1, 0, 1, 1, 1, 0, 0, -1], 5),
            ([-1, 0, 1, 0, 1, 1, 0, -1], 3),
            ([-1, 0, 1, 1, 0, 1, 0, -1], 4)
        ]

    def is_banned(self, pieces, enemy_pieces, pos):
        if self.chk_5(pieces, pos):
            return 0
        elif self.chk_overline(pieces, pos):
            return 1
        elif self.chk_44(pieces, enemy_pieces, pos):
            return 2
        elif self.chk_33(pieces, enemy_pieces, pos):
            return 3
        else:
            return 0
        
    def chk_5(self, pieces, pos):
        pieces[pos] = 1
        for dir in self.directions:
            if self.count_consecutive(pieces, pos, dir) == 5:
                pieces[pos] = 0
                return True
        pieces[pos] = 0
        return False
    
    def chk_overline(self, pieces, pos):
        pieces[pos] = 1
        for dir in self.directions:
            if self.count_consecutive(pieces, pos, dir) >= 6:
                pieces[pos] = 0
                return True
        pieces[pos] = 0
        return False
    
    def cnt_open3(self, pieces, enemy_pieces, pos, dir):
        dx, dy = dir
        count = 0
        skip = False
        
        for i, (pattern, gap_idx) in enumerate(self.open_three_patterns):
            if i == 1 and skip:
                continue
                
            match_positions = self.pattern_match(pieces, enemy_pieces, pos, dir, pattern)

            if len(match_positions) == 0:
                continue

            sx, sy = match_positions[0]
            gap_x, gap_y = sx + gap_idx*dx, sy + gap_idx*dy
            gap_pos_idx = gap_x + gap_y * self.board_width
                
            is_gap_banned = self.is_banned(pieces, enemy_pieces, gap_pos_idx) or self.chk_5(pieces, gap_pos_idx)
                
            if is_gap_banned:
                continue

            count += 1
            if i == 0:
                skip = True
        return count
    
    def cnt_4(self, pieces, enemy_pieces, pos, dir):
        count = 0
        skip = False

        for i, pattern in enumerate(self.fours):
            if i == 1 and skip:
                continue

            match_positions = self.pattern_match(pieces, enemy_pieces, pos, dir, pattern)

            if len(match_positions) > 0:
                count += len(match_positions)
                if i == 0:
                    skip = True

        return count
    
    def chk_44(self, pieces, enemy_pieces, pos):
        pieces[pos] = 1
        total_fours = 0
        for dir in self.directions:
            total_fours += self.cnt_4(pieces, enemy_pieces, pos, dir)
            if total_fours >= 2:
                pieces[pos] = 0
                return True
        pieces[pos] = 0
        return False
    
    def chk_33(self, pieces, enemy_pieces, pos):
        pieces[pos] = 1
        total_open3 = 0
        for dir in self.directions:
            total_open3 += self.cnt_open3(pieces, enemy_pieces, pos, dir)
            if total_open3 >= 2:
                pieces[pos] = 0
                return True
        pieces[pos] = 0
        return False
    
    def pattern_match(self, pieces, enemy_pieces, pos, dir, pattern):
        x, y = pos % self.board_width, pos // self.board_width
        dx, dy = dir
        is_match = True

        match_positions = []

        for offset in range(len(pattern)):
            sx = curr_x = x - dx * offset
            sy = curr_y = y - dy * offset
            is_match = True
            for p in pattern:
                if 0 <= curr_x < self.board_width and 0 <= curr_y < self.board_width:
                    if p == 1:
                        if pieces[curr_x + curr_y*self.board_width] != 1:
                            is_match = False
                            break
                    elif p == 0:
                        if pieces[curr_x + curr_y*self.board_width] == 1 or enemy_pieces[curr_x + curr_y*self.board_width] == 1:
                            is_match = False
                            break
                    elif p == -1:
                        if pieces[curr_x + curr_y*self.board_width] == 1:
                            is_match = False
                            break
                else:
                    if p != -1:
                        is_match = False
                        break
                curr_x, curr_y = curr_x + dx, curr_y + dy
            if is_match:
                match_positions.append((sx, sy))
        return match_positions

    def count_consecutive(self, pieces, pos, dir):
        x, y = pos % self.board_width, pos // self.board_width
        dx, dy = dir
        count = 0
        while True:
            if 0 <= x < self.board_width and 0 <= y < self.board_width:
                if pieces[x + y*self.board_width] == 1:
                    count += 1
                else: break
            else: break
            x, y = x + dx, y + dy

        x, y = pos % self.board_width, pos // self.board_width
        while True:
            x, y = x - dx, y - dy
            if 0 <= x < self.board_width and 0 <= y < self.board_width:
                if pieces[x + y*self.board_width] == 1:
                    count += 1
                else: break
            else: break
        return count
    
    def legal_actions(self, state):
        # 첫 수는 천원(중앙)에 두기
        if state.count_piece(state.pieces) == 0 and state.count_piece(state.enemy_pieces) == 0:
            return [ (self.board_width // 2) + (self.board_width // 2) * self.board_width ]
        actions = []
        for i in range(BOARD_WIDTH ** 2):
            if state.pieces[i] == 0 and state.enemy_pieces[i] == 0:
                actions.append(i)
        return actions
    
    def get_banned(self, state):
        ban_type = [0] * (self.board_width ** 2)
        if state.is_first_player():
            pieces = state.pieces
            enemy_pieces = state.enemy_pieces
            for pos in range(self.board_width ** 2):
                if pieces[pos] == 0 and enemy_pieces[pos] == 0:
                    ban_type[pos] = self.is_banned(pieces, enemy_pieces, pos)
        return ban_type