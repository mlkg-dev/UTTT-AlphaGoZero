from MCTS import MCTS
from UltimateTicTacToe import GAME_IN_PROGRESS, DRAW
from UltimateTicTacToe import UltimateTicTacToe
from NeuralNetwork import NeuralNetwork
import numpy as np
from keras import backend as K


class TestGame:

    def __init__(self, number_games_to_evaluate, old_network_path, new_network_path):
        self.agent_one = None
        self.agent_two = None
        self.stats = {"a1_wins": 0, "a2_wins": 0, "draws": 0}
        self.number_games_to_evaluate = number_games_to_evaluate
        self.old_network_path = old_network_path
        self.new_network_path = new_network_path

    def evaluate_agents(self):
        for i in range(0, self.number_games_to_evaluate):
            g1 = UltimateTicTacToe()
            g2 = UltimateTicTacToe()
            self.set_agent_one(g1, NeuralNetwork("Models/" + str(self.old_network_path)), 400)
            self.set_agent_two(g2, NeuralNetwork("Models/" + str(self.new_network_path)), 400)

            if i < 200:
                self.play(self.agent_one, self.agent_two, i)
            else:
                self.play(self.agent_two, self.agent_one, i)

            K.clear_session()

            print("Game " + str(i+1) + "/400")
            print("Result: " + "N1 " + str(self.stats["a1_wins"]) + " - " + str(self.stats["a2_wins"]) + " N2")
            print("Draws - " + str(self.stats["draws"]))

            if self.stats["a1_wins"] + self.stats["a2_wins"] > 0:
                print("New network - " + str((self.stats["a2_wins"] / (self.stats["a1_wins"] + self.stats["a2_wins"])) * 100) + "%")
                if self.stats["a2_wins"]/(self.stats["a1_wins"] + self.stats["a2_wins"]) > 0.55:
                    print("PASS")
                else:
                    print("FAIL")

    def play(self, agent_one, agent_two, game_number):
        agent_one_root = agent_one.root
        agent_two_root = agent_two.root

        agent_one.evaluate_first_state(agent_one_root)
        agent_two.evaluate_first_state(agent_two_root)

        while True:

            agent_one.do_playouts(agent_one_root)
            agent_one_root, pi = agent_one.choose_move(agent_one_root, True)
            agent_one_root.parent = None
            agent_one.add_dirichlet_noise_to_probabilities(agent_one_root)
            if agent_one_root.state[4] != GAME_IN_PROGRESS:
                if agent_one_root.state[4] == DRAW:
                    self.stats["draws"] += 1
                else:
                    if game_number < 200:
                        self.stats["a1_wins"] += 1
                    else:
                        self.stats["a2_wins"] += 1
                break

            agent_two_root = self.find_root_in_another_agent(agent_one_root, agent_two_root)

            agent_two.do_playouts(agent_two_root)
            agent_two_root, pi = agent_two.choose_move(agent_two_root, True)
            agent_two_root.parent = None
            agent_two.add_dirichlet_noise_to_probabilities(agent_two_root)

            if agent_two_root.state[4] != GAME_IN_PROGRESS:
                if agent_two_root.state[4] == DRAW:
                    self.stats["draws"] += 1
                else:
                    if game_number < 200:
                        self.stats["a2_wins"] += 1
                    else:
                        self.stats["a1_wins"] += 1
                break

            agent_one_root = self.find_root_in_another_agent(agent_two_root, agent_one_root)

    def set_agent_one(self, game, network, playouts):
        self.agent_one = MCTS(game, network, playouts)

    def set_agent_two(self, game, network, playouts):
        self.agent_two = MCTS(game, network, playouts)

    def find_root_in_another_agent(self, root, another_agent_root):
        raw_board = root.state[0]
        for child in another_agent_root.children:
            if np.array_equal(child.state[0], raw_board):
                return child
