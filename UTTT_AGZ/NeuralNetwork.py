from keras import models
from UltimateTicTacToe import CROSS,CIRCLE,EMPTY
from HelperFunctions import reshape_to_sudoku_matrix, update_index_after_reshape
import numpy as np


class NeuralNetwork:

    def __init__(self, file_path):
        self.model = models.load_model(file_path)

    def create_binary_layer(self, matrix, player):
        x_layer = np.copy(matrix)
        x_mask = x_layer == player
        return x_mask.astype(np.int8)

    def create_input_stack(self, state):
        player = (state[3] % 2) + 1

        matrix = np.copy(state[0])
        matrix = reshape_to_sudoku_matrix(matrix)

        if player == CROSS:
            player_layer = np.full((9, 9), 0)
        else:
            player_layer = np.full((9, 9), 1)

        current_move_x_layer = self.create_binary_layer(matrix, CROSS)
        current_move_o_layer = self.create_binary_layer(matrix, CIRCLE)

        current_move = update_index_after_reshape(state[2])

        matrix[current_move[0]][current_move[1]] = EMPTY

        previous_move_x_layer = self.create_binary_layer(matrix, CROSS)
        previous_move_o_layer = self.create_binary_layer(matrix, CIRCLE)

        input_stack = np.array([current_move_x_layer,
                               previous_move_x_layer,
                               current_move_o_layer,
                               previous_move_o_layer,
                               player_layer]
                               )
        return np.expand_dims(input_stack, axis=0)

    def convert_p_and_v(self, p, v):
        p = np.squeeze(p, axis=0)
        v = np.squeeze(v, axis=0)

        p.shape = (9, 9)
        p = reshape_to_sudoku_matrix(p)

        v = v[0]
        return p, v

    def evaluation(self, node_state):
        input_stack = self.create_input_stack(node_state)
        p, v = self.model.predict(input_stack, batch_size=1)
        p, v = self.convert_p_and_v(p, v)
        return p, v

    def first_evaluation(self, node_state):
        player_layer = np.full((9, 9), 1)
        current_move_x_layer = np.copy(node_state[0])
        current_move_o_layer = np.copy(node_state[0])
        previous_move_x_layer = np.copy(node_state[0])
        previous_move_o_layer = np.copy(node_state[0])
        input_stack = np.array([current_move_x_layer,
                                previous_move_x_layer,
                                current_move_o_layer,
                                previous_move_o_layer,
                                player_layer]
                               )
        input_stack = np.expand_dims(input_stack, axis=0)
        p, v = self.model.predict(input_stack, batch_size=1)
        p, v = self.convert_p_and_v(p, v)
        return p, v

