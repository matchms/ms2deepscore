from typing import List, Tuple

from tensorflow import keras

class BaseNetwork(keras.layers.Layer):
    def __init__(self, dims: Tuple[int, int, int] = (600, 600, 500), embedding_dim: int = 400,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.dense1 = keras.layers.Dense(dims[0],
                                         activation='relu',
                                         kernel_regularizer=keras.regularizers.l1(1e-6))
        self.dense2 = keras.layers.Dense(dims[1], activation='relu')
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dense3 = keras.layers.Dense(dims[2], activation='relu')
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.dense4 = keras.layers.Dense(embedding_dim, activation='relu')
        self.dropout3 = keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout1(x)
        x = self.dense3(x)
        x = self.dropout2(x)
        x = self.dense4(x)
        return self.dropout3(x)



class SiameseModel:
    def __init__(self,
                 input_shape: int,
                 base_dims: Tuple[int, int, int] = (600, 600, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.2):
        self.base = BaseNetwork(dims=base_dims,
                                embedding_dim=embedding_dim,
                                dropout_rate=dropout_rate)
        input_a = keras.layers.Input(shape=(1, input_shape), name="input_a")
        input_b = keras.layers.Input(shape=(1, input_shape), name="input_b")
        embedding_a = self.base(input_a)
        embedding_b = self.base(input_b)
        cosine_similarity = keras.layers.Dot(axes=(2, 2),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        self.model = keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity])

    def fit(self, x, y, validation_data, epochs):


