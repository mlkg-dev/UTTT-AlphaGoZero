import json
import pickle
import os, random
import numpy as np
from UltimateTicTacToe import CROSS,CIRCLE,EMPTY

#TODO - ADD JSON


class DataReader:

    def __init__(self):
        self.batch_size = 2048
        self.game_pool = len(os.listdir("Games/")) - 1 #.keep

    def read_json(self):
        pass

    def read_pickle(self):
        file_name = random.choice(os.listdir("Games/"))
        game = pickle.load(open("Games/" + file_name, 'rb'))
        return game

    def read_pickle_faster(self):
        random_number = random.randint(0, self.game_pool - 1)
        game = pickle.load(open("Games/" + "TRAINING_" + str(random_number) + ".UTTT", 'rb'))
        return game

    def make_minibatch(self, game, input_batch, output_batch):
        random_move = random.randint(0, len(game)-1)
        player = (random_move % 2) + 1

        matrix = np.copy(game[random_move]["state"])

        if player == CROSS:
            player_layer = np.full((9, 9), 0)
        else:
            player_layer = np.full((9, 9), 1)

        current_move_x_layer = self.create_binary_layer(matrix, CROSS)
        current_move_o_layer = self.create_binary_layer(matrix, CIRCLE)

        current_move = game[random_move]["move"]

        matrix[current_move[0]][current_move[1]] = EMPTY

        previous_move_x_layer = self.create_binary_layer(matrix, CROSS)
        previous_move_o_layer = self.create_binary_layer(matrix, CIRCLE)

        input_stack = np.array([current_move_x_layer,
                                previous_move_x_layer,
                                current_move_o_layer,
                                previous_move_o_layer,
                                player_layer]
                               )
        input_batch.append(input_stack)

        pi = []
        if random_move + 1 == len(game):
            pi = np.zeros(81)
        else:
            pi = game[random_move + 1]["pi"].flatten()

        output_batch["policy_head"].append(pi)
        output_batch["value_head"].append(game[random_move]["result"])

    def create_binary_layer(self, matrix, player):
        x_layer = np.copy(matrix)
        x_mask = x_layer == player
        return x_mask.astype(np.int8)

    def make_batch(self):
        input_batch = []
        output_batch = {"policy_head": [], "value_head": []}
        for i in range(self.batch_size):
            game = self.read_pickle_faster()
            self.make_minibatch(game, input_batch, output_batch)

        input_batch = np.array(input_batch)
        output_batch["policy_head"] = np.array(output_batch["policy_head"])
        output_batch["value_head"] = np.array(output_batch["value_head"])
        return input_batch, output_batch
