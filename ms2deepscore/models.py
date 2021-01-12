from typing import Tuple

from tensorflow import keras


class SiameseModel:
    """
    Class for training and evaluating a siamese neural network, implemented in Tensorflow Keras.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.

    Mimics keras.Model API.
    """
    def __init__(self,
                 input_dim: int,
                 base_dims: Tuple[int, int, int] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5):
        """
        Construct SiameseModel

        Parameters
        ----------
        input_dim
            Dimension of the input vector
        base_dims
            Size-3 tuple of integers depicting the dimensions of the 1st, 2nd, and 3rd hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model
        """
        self.base = self._get_base_model(input_dim=input_dim,
                                         dims=base_dims,
                                         embedding_dim=embedding_dim,
                                         dropout_rate=dropout_rate)
        input_a = keras.layers.Input(shape=input_dim, name="input_a")
        input_b = keras.layers.Input(shape=input_dim, name="input_b")
        embedding_a = self.base(input_a)
        embedding_b = self.base(input_b)
        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        self.model = keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity],
                                 name='head')

    @staticmethod
    def _get_base_model(input_dim: int,
                        dims: Tuple[int, int, int] = (600, 500, 500),
                        embedding_dim: int = 400,
                        dropout_rate: float = 0.25):
        model_input = keras.layers.Input(shape=input_dim, name='base_input')
        embedding = keras.layers.Dense(dims[0], activation='relu', name='dense1')(model_input)
        embedding = keras.layers.BatchNormalization(name='normalization1')(embedding)
        embedding = keras.layers.Dropout(dropout_rate, name='dropout1')(embedding)
        embedding = keras.layers.Dense(dims[1], activation='relu', name='dense2')(embedding)
        embedding = keras.layers.BatchNormalization(name='normalization2')(embedding)
        embedding = keras.layers.Dropout(dropout_rate, name='dropout2')(embedding)
        embedding = keras.layers.Dense(dims[2], activation='relu', name='dense3')(embedding)
        embedding = keras.layers.BatchNormalization(name='normalization3')(embedding)
        embedding = keras.layers.Dropout(dropout_rate, name='dropout3')(embedding)
        embedding = keras.layers.Dense(embedding_dim, activation='relu', name='embedding')(
            embedding)
        return keras.Model(model_input, embedding, name='base')

    def compile(self, *args, **kwargs):
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)

    def load_weights(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)

    def summary(self):
        self.base.summary()
        self.model.summary()

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)
