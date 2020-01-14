import json
import pickle
import numpy as np
from HelperFunctions import reshape_to_sudoku_matrix, update_index_after_reshape

'''
Save state - [0, 1, 2, 3]
0 - raw board representation (sudoku-like)
1 - 2 element list with last move position in matrix
2 - updated probability list (pi)
3 - result of the game (0 at first, updated after end of the game)
'''

class GameData:
    def __init__(self, save_method="pickle"):
        self.moves_list = []
        self.save_method = save_method
        self.game_number = pickle.load(open('save_config.config', 'rb'))

    def add_move_to_record(self, selected_move, pi):
        raw_board_state = np.copy(selected_move.state[0])
        raw_board_state = reshape_to_sudoku_matrix(raw_board_state)
        played_move = update_index_after_reshape(selected_move.state[2])
        self.moves_list.append([raw_board_state, played_move, pi, 0])

    def make_game_record(self, draw):
        if not draw:
            self.update_relative_result()
        self.save_to_file()

    def update_relative_result(self):
        result = 1
        for i in reversed(self.moves_list):
            i[3] = result
            result = -result

    def convert_data(self):
        dict_keys = [i for i in range(len(self.moves_list))]
        dict_values = [{'state': move[0], 'move': move[1], 'pi': move[2], 'result':move[3]} for move in self.moves_list]
        data = dict(zip(dict_keys, dict_values))
        return data

    def save_to_file(self):
        file_name = 'Games/' + 'TRAINING_' + str(self.game_number) + '.UTTT'
        if self.save_method == "pickle":
            self.save_with_pickle(file_name)
        if self.save_method == "json":
            self.save_with_json(file_name)
        self.game_number += 1
        pickle.dump(self.game_number, open('save_config.config', 'wb'))

    def save_with_json(self, file_name):
        data = self.convert_data()
        with open(file_name, 'w') as f:
            json.dump(data, f)
        f.close()

    def save_with_pickle(self, file_name):
        data = self.convert_data()
        pickle.dump(data, open(file_name, 'wb'))
