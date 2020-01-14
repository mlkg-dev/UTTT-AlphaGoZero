from keras import models


class NNHelperFunctions:

    def __init__(self):
        pass

    def save_weights(self, model, weights_file_path):
        model.save_weights(weights_file_path)

    def save_model(self, model,  file_path):
        models.save_model(model, file_path)
