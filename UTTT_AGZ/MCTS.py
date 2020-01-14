import math
from Node import Node
from UltimateTicTacToe import GAME_IN_PROGRESS, DRAW, CROSS, CIRCLE
from HelperFunctions import index_2d_to_1d_array, index_1d_to_2d_array
import numpy as np
import random


class MCTS:

    def __init__(self, game, network, playouts):
        self.game = game
        self.network = network
        self.playouts_number = playouts

        self.root = Node([np.copy(game.board), np.copy(game.meta_board), [-1, -1], 0, GAME_IN_PROGRESS], None, 0)

    def evaluate_first_state(self, root):
        p_root, v_root = self.network.first_evaluation(root.state)
        root.net_value = v_root
        root.visit_counter = 1
        root.evaluated = True
        expanded_root = self.game.expand_first_state()

        for state in expanded_root:
            self.set_probability_for_node(root, state, p_root)
        self.add_dirichlet_noise_to_probabilities(root)

    def MCTS_select(self, node):
        while True:
            max_node = [0, float("-inf")]
            for index, child in enumerate(node.children):
                selection_value = self.PUCT(child)
                if selection_value > max_node[1]:
                    max_node = [index, selection_value]

            if (not node.evaluated) or node.state[4] != GAME_IN_PROGRESS:
                return node
            node = node.children[max_node[0]]

    def MCTS_expansion_and_evaluation(self, node):
        node.evaluated = True

        if node.state[4] == GAME_IN_PROGRESS:
            p_node, v_node = self.network.evaluation(node.state)
            node.net_value = v_node
            expanded_node = self.game.expand(node.state)
            for state in expanded_node:
                self.set_probability_for_node(node, state, p_node)
        else:

            if node.state[4] == DRAW:
                node.net_value = 0
            else:
                node.net_value = 1.0
                node.policy = 1.0
                node.q_policy = 1.0

    def MCTS_backpropagation(self, node):
        last_node_evaluated_net_value = node.net_value

        node.visit_counter += 1
        node.sum_net_value = node.net_value
        node.q_policy = node.sum_net_value / node.visit_counter

        while True:
            if node.parent is not None:
                node = node.parent
                node.visit_counter += 1
                last_node_evaluated_net_value = -last_node_evaluated_net_value
                node.sum_net_value += last_node_evaluated_net_value
                node.q_policy = node.sum_net_value / node.visit_counter
            else:
                break

    def do_playouts(self, root):
        playout_counter = 0
        while playout_counter < self.playouts_number:
                playout_node = root
                playout_node = self.MCTS_select(playout_node)
                self.MCTS_expansion_and_evaluation(playout_node)
                self.MCTS_backpropagation(playout_node)
                playout_counter += 1

    def choose_move(self, node, self_play=False):
        pi_probability_distribution = np.zeros(81)
        children_with_1d_index = []
        for child in node.children:
            index_1d = index_2d_to_1d_array(child.state[2][0], child.state[2][1])
            pi_probability_distribution[index_1d] = child.visit_counter / node.visit_counter
            children_with_1d_index.append([index_1d, child])

        if self_play and node.state[3] < 8:
            semi_random_move_index = []
            while not semi_random_move_index:
                pi_probability_distribution /= pi_probability_distribution.sum()
                semi_random_move_index = np.random.choice(range(len(pi_probability_distribution)), p=pi_probability_distribution)
            for i in children_with_1d_index:
                if i[0] == semi_random_move_index:
                    return [i[1], pi_probability_distribution]

        else:
            best_node = [0, 0]
            for child in node.children:
                if child.visit_counter > best_node[1]:
                    best_node = [child, child.visit_counter]
            return [best_node[0], pi_probability_distribution]

    def PUCT(self, node):
        c = 4
        return node.q_policy + c * node.policy * (math.sqrt(node.parent.visit_counter) / (1 + node.visit_counter))

    def set_probability_for_node(self, node, state, p_node):
        state_probability = p_node[state[2][0]][state[2][1]]
        node.add_child(state, state_probability)

    def add_dirichlet_noise_to_probabilities(self, node):
        epsilon = 0.25
        alpha = 2
        n = len(node.children)

        dirichlet_distribution = np.random.dirichlet(n*[alpha])

        for dirichlet_index, child in enumerate(node.children):
            child.policy = (1-epsilon) * child.policy + epsilon * dirichlet_distribution[dirichlet_index]

    def print_output(self, node):
        self.game.print_board(node.state[0])
        print("Last move = "+str(node.state[2]))
        print("Policy = " + str(node.policy) + " Q_policy = " + str(node.q_policy))
        print("NetValue = " + str(node.net_value) + " Visits = " + str(node.visit_counter))
        print("MetaBoard = "+str(node.state[1]))
        self.game.print_metaboard(node.state[1])
        self.print_visits_for_children(node.parent)
        print(40 * "-")

    def print_result(self, result):
        if result == CROSS:
            print("X won"+"("+str(CROSS)+")")
        if result == CIRCLE:
            print("O won"+"("+str(CIRCLE)+")")
        if result == DRAW:
            print("Draw"+"("+str(DRAW)+")")

    def print_visits_for_children(self, node):
        print("Children:")
        visits_list = []
        for child in node.children:
            visits_list.append(["visits=" + str(child.visit_counter)])
        print(visits_list)
