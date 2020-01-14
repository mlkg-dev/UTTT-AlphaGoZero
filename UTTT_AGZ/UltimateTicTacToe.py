import numpy as np

EMPTY = 0
GAME_IN_PROGRESS = 0
CROSS = 1
CIRCLE = 2
DRAW = 3

'''
Game state list - [0, 1 , 2 , 3, 4]
0 - raw board matrix
1 - raw meta board matrix
2 - 2 element list with last move position in matrix
3 - move number
4 - result of the game
'''

class UltimateTicTacToe:

    def __init__(self):
        self.meta_board = np.zeros(9, dtype=np.int8)
        self.board = np.zeros((9, 9), dtype=np.int8)

    def expand_first_state(self):
        tmp_board = np.copy(self.board)
        tmp_meta_board = np.copy(self.meta_board)
        expanded_list = []

        for i in range(0, 9):
            for j in range(0, 9):
                tmp_board[i][j] = CROSS
                expanded_list.append([np.copy(tmp_board), np.copy(tmp_meta_board), [i, j], 1, GAME_IN_PROGRESS])
                tmp_board[i][j] = EMPTY

        return expanded_list

    def expand(self, state):
        expanded_list = []
        tmp_board = np.copy(state[0])
        tmp_meta_board = np.copy(state[1])
        current_move = state[2][1]
        player = (state[3] % 2) + 1
        move_number = state[3] + 1

        if state[1][current_move] != EMPTY:
            for i in range(0, 9):
                if state[1][i] == EMPTY:
                    for j in range(0, 9):
                        if state[0][i][j] == EMPTY:
                            tmp_board[i][j] = player
                            game_result = self.check_small_square(tmp_board, tmp_meta_board, i)
                            expanded_list.append([np.copy(tmp_board), np.copy(tmp_meta_board), [i, j], move_number, game_result])
                            tmp_board[i][j] = EMPTY
                            tmp_meta_board = np.copy(state[1])

        else:
            for j in range(0, 9):
                if state[0][current_move][j] == EMPTY:
                    tmp_board[current_move][j] = player
                    game_result = self.check_small_square(tmp_board, tmp_meta_board, current_move)
                    expanded_list.append([np.copy(tmp_board), np.copy(tmp_meta_board), [current_move, j], move_number, game_result])
                    tmp_board[current_move][j] = EMPTY
                    tmp_meta_board = np.copy(state[1])

        return expanded_list

    def check_small_square(self, tmp_board, tmp_meta_board, i):
        small_square = tmp_board[i]
        for j in range(0, 9, 3):
            if small_square[j] != EMPTY and (small_square[j] == small_square[j + 1] == small_square[j + 2]):
                tmp_meta_board[i] = small_square[j]
                return self.check_game_over(tmp_meta_board)
        for j in range(0, 3):
            if small_square[j] != EMPTY and (small_square[j] == small_square[j + 3] == small_square[j + 6]):
                tmp_meta_board[i] = small_square[j]
                return self.check_game_over(tmp_meta_board)
        if small_square[4] != EMPTY and ((small_square[0] == small_square[4] == small_square[8])
                                     or (small_square[2] == small_square[4] == small_square[6])):
            tmp_meta_board[i] = small_square[4]
            return self.check_game_over(tmp_meta_board)

        if EMPTY not in small_square:
            tmp_meta_board[i] = DRAW

        return self.check_draw(tmp_meta_board)

    def check_draw(self, tmp_meta_board):
        if EMPTY not in tmp_meta_board:
            return DRAW
        else:
            return GAME_IN_PROGRESS

    def check_game_over(self, tmp_meta_board):
        for i in range(0, 9, 3):
            if tmp_meta_board[i] != EMPTY and (tmp_meta_board[i] == tmp_meta_board[i + 1] == tmp_meta_board[i + 2]):
                return tmp_meta_board[i]

        for i in range(0, 3):
            if tmp_meta_board[i] != EMPTY and (tmp_meta_board[i] == tmp_meta_board[i + 3] == tmp_meta_board[i + 6]):
                return tmp_meta_board[i]

        if tmp_meta_board[4] != EMPTY and (((tmp_meta_board[0] == tmp_meta_board[4] == tmp_meta_board[8]) or
                                        (tmp_meta_board[2] == tmp_meta_board[4] == tmp_meta_board[6]))):
            return tmp_meta_board[4]

        return self.check_draw(tmp_meta_board)

    def print_board(self, board):
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                print("|" + str(board[i][j:j + 3]) + "|" + str(board[i + 1][j:j + 3]) + "|" + str(
                    board[i + 2][j:j + 3]) + "|")
            print("|" + '-' * 7 + "|" + '-' * 7 + "|" + '-' * 7 + "|")

    def print_metaboard(self, meta_board):
        for i in range(0, 9, 3):
            print("||" + str(meta_board[i:i+3])+"||")
