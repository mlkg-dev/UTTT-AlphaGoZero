from keras import models
from DataReader import DataReader
from NNHelperFunctions import NNHelperFunctions
import pickle


class NetworkTraining:

    def __init__(self, file_path):
        self.model = models.load_model(file_path)
        self.data_reader = DataReader()
        self.nn_helper_funcs = NNHelperFunctions()
        self.history = []

    def train_network(self):
        for counter in range(1000):
            input_batch, output_batch = self.data_reader.make_batch()
            print(str(counter+1) + "/" + str(1000) + " batch ready")
            epoch_history = self.model.fit(input_batch, output_batch, batch_size=2048)
            self.history.append(epoch_history)
            print(str(counter+1) + "/" + str(1000) + " iteration trained")
        print("DONE")

    def save_after_training(self):
        self.nn_helper_funcs.save_model(self.model, "Models/new_network.h5")
        self.nn_helper_funcs.save_weights(self.model, "Models/new_network_weights.h5")

    def save_history(self):
        pickle.dump(self.history, open("Models/history.history", 'wb'))
