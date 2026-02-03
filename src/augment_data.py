import numpy as np

def augment_data(game : int, num_moves : int) -> None:
    for turn in range(1, num_moves):
        board_string = ""
        move_prob = ""

        with open(f"games/game{game}/turn{turn}.txt","r") as file:
            # get board
            for i in range(17):
                board_string+=(next(file))
            
            # get policy
            while("]" not in move_prob):
                move_prob += next(file)
            
            # get value
            win = float(next(file))

            # process board
            lines = board_string.strip().split("\n")
            board = []
            for line in lines:
                if "|" not in line:
                    row = line.split(" - ")
                    for i in range(len(row)):
                        if(row[i] == "O"): row[i] = 0
                        elif(row[i] == "B"): row[i] = 1
                        else: row[i] = 2
                    board.append(row)

            # format policies
            move_prob = move_prob.strip().strip("[]").split(",")
            for i in range(len(move_prob)):
                move_prob[i] = float(move_prob[i])
            
            # symmetry based transformations on board
            r_board = np.rot90(board, 1)
            rr_board = np.rot90(board, 2)
            rrr_board = np.rot90(board, 3)
            hf_board = np.fliplr(board)
            vf_board = np.flipud(board)

            # symmetry based transformations on policy
            np_move_prob = np.array(move_prob)
            move_prob_2d = np_move_prob.reshape(9, 9)

            r_move_prob_2d = np.rot90(move_prob_2d, 1)
            r_move_prob = r_move_prob_2d.reshape(-1).tolist()
            rr_move_prob_2d = np.rot90(move_prob_2d, 2)
            rr_move_prob = rr_move_prob_2d.reshape(-1).tolist()
            rrr_move_prob_2d = np.rot90(move_prob_2d, 3)
            rrr_move_prob = rrr_move_prob_2d.reshape(-1).tolist()
            hf_move_prob_2d = np.fliplr(move_prob_2d)
            hf_move_prob = hf_move_prob_2d.reshape(-1).tolist()
            vf_move_prob_2d = np.flipud(move_prob_2d)
            vf_move_prob = vf_move_prob_2d.reshape(-1).tolist()

            # write to files
            augment_write(game, turn, "r", r_board, r_move_prob, win)
            augment_write(game, turn, "rr", rr_board, rr_move_prob, win)
            augment_write(game, turn, "rrr", rrr_board, rrr_move_prob, win)
            augment_write(game, turn, "hf", hf_board, hf_move_prob, win)
            augment_write(game, turn, "vf", vf_board, vf_move_prob, win)


def augment_write(game : int, turn : int, aug_type : str, 
                  board : list[list[int]], move_prob : list[float], 
                  win : float) -> None:
    with open(f"games/game{game}{aug_type}/turn{turn}.txt","w") as file:
        str_board = ""

        # convert borad fron int to strings
        for i in range(9):
            for j in range(9):
                if(board[i][j] == 0):
                    str_board += "O"
                elif(board[i][j] == 1):
                    str_board += "B"
                else:
                    str_board += "W"
                
                if (j != 9 - 1):
                    str_board+=" - "
                else:
                    str_board+="\n"
                    if (i != 9 - 1):
                        str_board += "|   " * 9
            if i != 9 - 1:
                str_board+= "\n"

        # write to file
        file.write(str_board)
        file.write(str(move_prob))
        file.write("\n")
        file.write(str(win))