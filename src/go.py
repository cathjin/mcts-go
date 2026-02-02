class Go:
    def __init__(self, dim = 19):
        self.dim = dim
        self.board = [["O"]*dim for _ in range(dim)]
        self.curr_player = "B"
        self.turn_number = 0

    def print_board(self):
        str_board = ""
        for i in range(self.dim):
            for j in range(self.dim):
                str_board += self.board[i][j]
                if (j != self.dim - 1):
                    str_board+=" - "
                else:
                    str_board+="\n"
                    if (i != self.dim - 1):
                        str_board += "|   " * self.dim
            if i != self.dim - 1:
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
            # self.print_board()
            self.curr_player = "W"
        else:
            # move_x = input("White's Move Row: ")
            # move_y = input("White's Move Column: ")
            self.board[int(move_x)][int(move_y)] = "W"
            self.check_all_captured("B", "W")
            # self.print_board()
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
        checked = []
        for x,y in indices:
            if((x,y) in checked): 
                continue
            if self.check_captured(player, opp, x,y, checked):
                # captured.append((x,y))
                self.board[x][y] = "O"
            
        # for x,y in captured:
        #     self.board[x][y] = "O"
        return captured

    def check_captured(self, player, opp, x, y, checked):
        neighbour_indices = self.get_neighbours_indices(x,y)
        neighbour_symbols = self.get_neighbours_symbol(neighbour_indices)
        checked.append((x,y))

        for i in range(len(neighbour_symbols)):
            if neighbour_symbols[i] == player and neighbour_indices[i] not in checked:
                checked.append(neighbour_indices[i])

        if "O" in neighbour_symbols:
            return False 
        elif player in neighbour_symbols:
            result = True
            for i in range(len(neighbour_symbols)):
                if neighbour_symbols[i] == player and neighbour_indices[i] not in checked:
                    x1, y1 = neighbour_indices[i]
                    result = result and self.check_captured(player, opp, x1, y1, checked)
            return result
        else:
            return True

    def get_neighbours_indices(self, x, y):
        if x== 0 and y== 0:
            return [(1, 0), (0, 1)] 
        elif x == self.dim - 1 and y== self.dim - 1:
            return [(self.dim-2, self.dim-1), (self.dim-1, self.dim - 2)]
        elif x == 0 and y == self.dim-1: 
            return [(1, self.dim-1), (0, self.dim-2)]
        elif x == self.dim-1 and y== 0:
            return[(self.dim - 2, 0), (self.dim-1, 1)]
        elif x == self.dim -1:
            return[(self.dim - 2, y), (self.dim-1, y-1), (self.dim-1, y + 1)]
        elif y == self.dim -1:
            return[(x - 1, self.dim-1), (x, self.dim-2), (x + 1, self.dim-1)]
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
        
        for i in range(self.dim - 1):
            for j in range(self.dim - 1):
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
