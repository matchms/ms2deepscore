from pathlib import Path
from typing import Tuple, Union
import h5py
import json
from tensorflow import keras
from tensorflow.python.keras.saving import hdf5_format

from ms2deepscore import SpectrumBinner


def load_model(filename: Union[str, Path]):
    """
    Load a MS2DeepScore model (SiameseModel) from file.
    
    Parameters
    ----------
    filename
        Filename. Expecting saved SiameseModel.
        
    """
    with h5py.File(filename, mode='r') as f:
        binner_json = f.attrs['spectrum_binner']
        keras_model = hdf5_format.load_model_from_hdf5(f)

    # Reconstitute spectrum_binner
    binner_dict = json.loads(binner_json)
    spectrum_binner = SpectrumBinner(binner_dict["number_of_bins"],
                                     binner_dict["mz_max"], binner_dict["mz_min"],
                                     binner_dict["peak_scaling"],
                                     binner_dict["allowed_missing_percentage"])
    spectrum_binner.peak_to_position = {int(key): value for key, value in binner_dict["peak_to_position"].items()}
    spectrum_binner.known_bins = binner_dict["known_bins"]

    # Extract parameters for SiameseModel
    embedding_dim = keras_model.layers[2].output_shape[1]
    base_dims = []
    for layer in keras_model.layers[2].layers:
        if "dense" in layer.name:
            base_dims.append(layer.output_shape[1])
        elif "dropout" in layer.name:
            dropout_rate = layer.rate

    model = SiameseModel(spectrum_binner, base_dims, embedding_dim, dropout_rate)
    # TODO: Now this creates a keras model in the SiameseModel.__init__() and then replaces this. Seems unefficient.
    model.base = keras_model.layers[2]
    model.model = keras_model
    return model
    

class SiameseModel:
    """
    Class for training and evaluating a siamese neural network, implemented in Tensorflow Keras.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.

    Mimics keras.Model API.
    """
    def __init__(self,
                 spectrum_binner: SpectrumBinner,
                 base_dims: Tuple[int, int, int] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5):
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
        """
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.input_dim = len(spectrum_binner.known_bins)
        self.base = self._get_base_model(input_dim=self.input_dim,
                                         dims=base_dims,
                                         embedding_dim=embedding_dim,
                                         dropout_rate=dropout_rate)
        input_a = keras.layers.Input(shape=self.input_dim, name="input_a")
        input_b = keras.layers.Input(shape=self.input_dim, name="input_b")
        embedding_a = self.base(input_a)
        embedding_b = self.base(input_b)
        cosine_similarity = keras.layers.Dot(axes=(1, 1),
                                             normalize=True,
                                             name="cosine_similarity")([embedding_a, embedding_b])
        self.model = keras.Model(inputs=[input_a, input_b], outputs=[cosine_similarity],
                                 name='head')

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.

        Parameters
        ----------
        filename
            Filename to specify where to store the model.

        """
        binner_dict = self.spectrum_binner.__dict__
        binner_json = json.dumps(binner_dict)
    
        # Save model
        with h5py.File(filename, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.model, f)
            f.attrs['spectrum_binner'] = binner_json

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
