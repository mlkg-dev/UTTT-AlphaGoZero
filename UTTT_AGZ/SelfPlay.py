from MCTS import MCTS
from GameData import GameData
from UltimateTicTacToe import GAME_IN_PROGRESS, DRAW


class SelfPlay:

    def __init__(self, game, network, playouts):
        self.mcts = MCTS(game, network, playouts)
        self.gd = GameData()

    def play(self):
        root = self.mcts.root
        self.mcts.evaluate_first_state(root)

        while root.state[4] == GAME_IN_PROGRESS:
            self.mcts.do_playouts(root)
            root, pi = self.mcts.choose_move(root, True)
            self.gd.add_move_to_record(root, pi)
            root.parent = None
            self.mcts.add_dirichlet_noise_to_probabilities(root)

        self.mcts.print_result(root.state[4])
        self.gd.make_game_record(True) if root.state[4] == DRAW else self.gd.make_game_record(False)


