NUM_COLS = 9
NUM_ROWS = 9

BOARD = [["O"]*NUM_COLS for _ in range(NUM_ROWS)]

def print_board():
    for i in range(NUM_ROWS):
        for j in range(NUM_COLS):
            print(BOARD[i][j], end="")
            if (j != NUM_COLS - 1):
                print(" - ", end = "")
            else:
                print()
                if (i != NUM_ROWS - 1):
                    print("|   " * NUM_COLS)

def play_game():
    while(True):
        move_x = input("Black's Move Row: ")
        move_y = input("Black's Move Column: ")
        BOARD[int(move_x) - 1][int(move_y) - 1] = "B"
        check_all_captured("W", "B")
        print_board()


        move_x = input("White's Move Row: ")
        move_y = input("White's Move Column: ")
        BOARD[int(move_x) - 1][int(move_y) - 1] = "W"
        check_all_captured("B", "W")
        print_board()

def check_all_captured(player, opp):
    indices = []
    for i, row in enumerate(BOARD):
        for j, element in enumerate(row):
            if element == player:
                indices.append((i, j))  # row, col
    captured = []
    for x,y in indices:
        if check_captured(player, opp, x,y):
            captured.append((x,y))
    print("Captured", captured)
    for x,y in captured:
        BOARD[x][y] = "O"
    return captured

def check_captured(player, opp, x, y, checked = []):
    neighbour_indices = get_neighbours_indices(x,y,player)
    neighbour_symbols = get_neighbours_symbol(neighbour_indices)
    if "O" in neighbour_symbols:
        return False
    else:
        for i in range(len(neighbour_symbols)):
            if neighbour_symbols[i] == player and neighbour_indices[i] not in checked:
                x1, y1 = neighbour_indices[i]
                checked.append((x,y))
                return check_captured(player, opp, x1, y1)
        return True



def get_neighbours_indices(x, y, player):
    if x== 0 and y== 0:
        return [(1, 0), (0, 1)] 
    elif x == NUM_ROWS - 1 and y== NUM_COLS - 1:
        return [(NUM_ROWS-2, NUM_COLS-1), (NUM_ROWS-1, NUM_COLS - 2)]
    elif x == 0 and y == NUM_COLS-1:
        return [(1, NUM_COLS-1), (0, NUM_COLS-2)]
    elif x == NUM_ROWS-1 and y== 0:
        return[(NUM_ROWS - 2, 0), (NUM_ROWS-1, 1)]
    elif x == NUM_ROWS -1:
        return[(NUM_ROWS - 2, y), (NUM_ROWS-1, y-1), (NUM_ROWS-1, y + 1)]
    elif y == NUM_COLS -1:
        return[(x - 1, NUM_COLS-1), (x, NUM_COLS-2), (x + 1, NUM_COLS-1)]
    else:
        return[(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def get_neighbours_symbol(neighbour_indices):
    symbols = []
    for x, y in neighbour_indices:
        print(x,y)
        symbols.append(BOARD[x][y])
    return symbols


play_game()