class Go:
    def __init__(self, num_rows = 19, num_cols = 19):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.board = [["O"]*num_cols for _ in range(num_rows)]
        self.prev_state = [["O"]*num_cols for _ in range(num_rows)]
        self.curr_player = "B"
        self.turn_number = 0

    def print_board(self):
        str_board = ""
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                str_board += self.board[i][j]
                if (j != self.num_cols - 1):
                    str_board+=" - "
                else:
                    str_board+="\n"
                    if (i != self.num_rows - 1):
                        str_board += "|   " * self.num_cols
            if i != self.num_rows - 1:
                str_board+= "\n"
        return str_board

    def play_move(self, player, move_x, move_y):
        # while(True):
            # self.prev_board = self.board # need to make deep copy

        if(player == "B"):
            # move_x = input("Black's Move Row: ")
            # move_y = input("Black's Move Column: ")
            # self.board[9 - int(move_x)][ord(move_y) - ord('A')] = "B"
            self.board[int(move_x)][int(move_y)] = "B"
            self.check_all_captured("W", "B")
            self.print_board()
            self.curr_player = "W"
        else:
            # move_x = input("White's Move Row: ")
            # move_y = input("White's Move Column: ")
            self.board[int(move_x)][int(move_y)] = "W"
            self.check_all_captured("B", "W")
            self.print_board()
            self.curr_player = "B"

        # if self.prev_board == self.board:
            # break # change later


    def check_all_captured(self, player, opp):
        indices = []
        for i, row in enumerate(self.board):
            for j, element in enumerate(row):
                if element == player:
                    indices.append((i, j))  # row, col
        captured = []
        for x,y in indices:
            checked = []
            if self.check_captured(player, opp, x,y, checked):
                captured.append((x,y))
            
        for x,y in captured:
            self.board[x][y] = "O"
        return captured

    # needs reworking
    def check_captured(self, player, opp, x, y, checked):
        neighbour_indices = self.get_neighbours_indices(x,y)
        neighbour_symbols = self.get_neighbours_symbol(neighbour_indices)
        if "O" in neighbour_symbols:
            return False 
        elif player in neighbour_symbols:
            checked.append((x,y))
            result = True
            for i in range(len(neighbour_symbols)):
                if neighbour_symbols[i] == player and neighbour_indices[i] not in checked:
                    x1, y1 = neighbour_indices[i]
                    result = result and self.check_captured(player, opp, x1, y1, checked)
            return result
        else:
            return True



    def get_neighbours_indices(self, x, y):
        if x== 0 and y== 0: # (0,0)
            return [(1, 0), (0, 1)] 
        elif x == self.num_rows - 1 and y== self.num_cols - 1:
            return [(self.num_rows-2, self.num_cols-1), (self.num_rows-1, self.num_cols - 2)]
        elif x == 0 and y == self.num_cols-1: 
            return [(1, self.num_cols-1), (0, self.num_cols-2)]
        elif x == self.num_rows-1 and y== 0:
            return[(self.num_rows - 2, 0), (self.num_rows-1, 1)]
        elif x == self.num_rows -1:
            return[(self.num_rows - 2, y), (self.num_rows-1, y-1), (self.num_rows-1, y + 1)]
        elif y == self.num_cols -1:
            return[(x - 1, self.num_cols-1), (x, self.num_cols-2), (x + 1, self.num_cols-1)]
        elif x == 0:
            return[(0, y - 1), (0, y + 1), (1, y)]
        elif y == 0:
            return[(x - 1, 0), (x + 1, 0), (x, 1)]
        else:
            return[(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

    def get_neighbours_symbol(self, neighbour_indices):
        symbols = []
        for x, y in neighbour_indices:
            symbols.append(self.board[x][y])
        return symbols

    def score(self):
        black_score = 0
        white_score = 0

        # count num stones
        checked = set()

        def score_area(x, y):
            to_visit = [(x,y)]
            points = 0
            symbols = set()
            
            while to_visit:
                check_x, check_y = to_visit.pop()
                if (check_x, check_y) in checked:
                    continue
                checked.add((check_x, check_y))
                points += 1
                neighbour_indices = self.get_neighbours_indices(check_x, check_y)
                neighbour_symbols = self.get_neighbours_symbol(neighbour_indices)
                for i in range(len(neighbour_symbols)):
                    if neighbour_symbols[i] == "O":
                        to_visit.append(neighbour_indices[i])
                    else:
                        symbols.add(neighbour_symbols[i])
            return points, symbols
        
        for i in range(self.num_rows - 1):
            for j in range(self.num_cols - 1):
                if self.board[i][j] == "B":
                    black_score += 1
                elif self.board[i][j] == "W":
                    white_score += 1
                else:
                    count, symbols = score_area(i, j)
                    if len(symbols) == 2:
                        continue
                    elif "B" in symbols:
                        black_score += count
                    elif "W" in symbols:
                        white_score += count
        return white_score, black_score # change later
