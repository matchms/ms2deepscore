from pathlib import Path
from typing import Tuple, Union
import h5py
from tensorflow import keras
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore import SpectrumBinner


class SiameseModel:
    """
    Class for training and evaluating a siamese neural network, implemented in Tensorflow Keras.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.

    Mimics keras.Model API.

    For example:

    .. code-block:: python

        # Import data and reference scores --> spectrums & tanimoto_scores_df

        # Create binned spectrums
        spectrum_binner = SpectrumBinner(1000, mz_min=10.0, mz_max=1000.0, peak_scaling=0.5)
        binned_spectrums = spectrum_binner.fit_transform(spectrums)

        # Create generator
        dimension = len(spectrum_binner.known_bins)
        test_generator = DataGeneratorAllSpectrums(binned_spectrums, tanimoto_scores_df,
                                                   dim=dimension)

        # Create (and train) a Siamese model
        model = SiameseModel(spectrum_binner, base_dims=(600, 500, 400), embedding_dim=400,
                             dropout_rate=0.2)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))
        model.summary()
        model.fit(test_generator,
                  validation_data=test_generator,
                  epochs=50)

    """
    def __init__(self,
                 spectrum_binner: SpectrumBinner,
                 base_dims: Tuple[int, int, int] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5,
                 keras_model: keras.Model = None):
        """
        Construct SiameseModel

        Parameters
        ----------
        spectrum_binner
            SpectrumBinner which is used to bin the spectra data for the model training.
        base_dims
            Size-3 tuple of integers depicting the dimensions of the 1st, 2nd, and 3rd hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model
        keras_model
            When provided, this keras model will be used to construct the SiameseModel instance.
            Default is None.
        """
        # pylint: disable=too-many-arguments
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.input_dim = len(spectrum_binner.known_bins)

        if keras_model is None:
            # Create base model
            self.base = self._get_base_model(input_dim=self.input_dim,
                                             dims=base_dims,
                                             embedding_dim=embedding_dim,
                                             dropout_rate=dropout_rate)
            # Create head model
            self.model = self._get_head_model(input_dim=self.input_dim,
                                              base_model=self.base)
        else:
            self._construct_from_keras_model(keras_model)

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.

        Parameters
        ----------
        filename
            Filename to specify where to store the model.

        """
        with h5py.File(filename, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.model, f)
            f.attrs['spectrum_binner'] = self.spectrum_binner.to_json()

    @staticmethod
    def _get_base_model(input_dim: int,
                        dims: Tuple[int, int, int] = (600, 500, 500),
                        embedding_dim: int = 400,
                        dropout_rate: float = 0.25):
        model_input = keras.layers.Input(shape=input_dim, name='base_input')
        embedding = keras.layers.Dense(dims[0], activation='relu', name='dense1',
                                       kernel_regularizer=keras.regularizers.l1_l2(l1=1e-6, l2=1e-6))(model_input)
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

    @staticmethod
    def _get_head_model(input_dim: int,
                        base_model: keras.Model):
        input_a = keras.layers.Input(shape=input_dim, name="input_a")
        input_b = keras.layers.Input(shape=input_dim, name="input_b")
        embedding_a = base_model(input_a)
        embedding_b = base_model(input_b)
        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        return keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity],
                           name='head')

    def _construct_from_keras_model(self, keras_model):
        assert isinstance(keras_model, keras.Model), "Expected keras model as input."
        self.base = keras_model.layers[2]
        self.model = keras_model

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
