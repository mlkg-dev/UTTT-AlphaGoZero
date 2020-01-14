import time
from UltimateTicTacToe import UltimateTicTacToe
from NeuralNetwork import NeuralNetwork
from SelfPlay import SelfPlay
from keras import backend as K

game_counter = 0
n = NeuralNetwork("Models/current_network.h5")

while True:
    start = time.time()
    g = UltimateTicTacToe()
    sp = SelfPlay(g, n, 400)
    sp.play()
    g = None
    sp = None
    K.clear_session()
    end = time.time()
    print("Game generated in " + str(end - start) + "s")
    game_counter += 1
    print("Games played = " + str(game_counter))
