from keras import layers, backend, models, regularizers
from keras.optimizers import SGD


class NeuralNetworkModel:

    def __init__(self):
        self.model = None
        self.residual_blocks_number = 4
        self.input_shape = (5, 9, 9)    
        self.filters = 32
        self.kernel_size = (3, 3)
        self.regularization_const = 10**-4
        self.learning_rate = 0.01
        self.momentum = 0.9

    def create_convolution_block(self, previous_block):
        convolution_block = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            activation="linear",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const)
        )(previous_block)

        convolution_block = layers.BatchNormalization(axis=1)(convolution_block)
        convolution_block = layers.ReLU()(convolution_block)

        return convolution_block

    def create_residual_block(self, previous_block):
        residual_block = self.create_convolution_block(previous_block)
        residual_block = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            data_format="channels_last",
            activation="linear",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const)
        )(residual_block)

        residual_block = layers.BatchNormalization(axis=1)(residual_block)
        residual_block = layers.add([previous_block, residual_block])
        residual_block = layers.ReLU()(residual_block)
        return residual_block

    def create_policy_head(self, previous_block):
        policy_block = layers.Conv2D(
            filters=2,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            data_format="channels_last",
            activation="linear",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const)
        )(previous_block)
        policy_block = layers.BatchNormalization(axis=1)(policy_block)
        policy_block = layers.ReLU()(policy_block)
        policy_block = layers.Flatten()(policy_block)
        policy_block = layers.Dense(
            units=self.input_shape[1] * self.input_shape[2],
            activation="softmax",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const),
            name="policy_head"
        )(policy_block)
        return policy_block

    def create_value_head(self, previous_block):
        value_block = layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            data_format="channels_last",
            activation="linear",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const)
        )(previous_block)
        value_block = layers.BatchNormalization(axis=1)(value_block)
        value_block = layers.ReLU()(value_block)
        value_block = layers.Flatten()(value_block)
        value_block = layers.Dense(
            units=256,
            activation="linear",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const),
        )(value_block)
        value_block = layers.ReLU()(value_block)
        value_block = layers.Dense(
            units=1,
            activation="tanh",
            use_bias=False,
            kernel_regularizer=regularizers.l2(l=self.regularization_const),
            name="value_head"
        )(value_block)
        return value_block

    def create_model(self):
        network_input = layers.Input(shape=self.input_shape)

        block = self.create_convolution_block(network_input)

        for i in range(self.residual_blocks_number):
            block = self.create_residual_block(block)

        policy_head = self.create_policy_head(block)

        value_head = self.create_value_head(block)

        self.model = models.Model(inputs=[network_input], outputs=[policy_head, value_head])
        self.model.compile(
            optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),
            loss={"policy_head": "categorical_crossentropy", "value_head": "mse"},
            metrics={"policy_head": "acc", "value_head": "mse"},
            loss_weights={"policy_head": 0.5, "value_head": 0.5}
        )

    def save_model(self, filepath):
        models.save_model(self.model, filepath)
        self.model.summary()


nnm = NeuralNetworkModel()
f_path = "Models/network.h5"
nnm.create_model()
nnm.save_model(f_path)
print("DONE")
