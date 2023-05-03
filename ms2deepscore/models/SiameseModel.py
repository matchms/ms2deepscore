from pathlib import Path
from typing import Tuple, Union
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from ms2deepscore import SpectrumBinner


class SiameseModel(nn.Module):
    """
    Class for training and evaluating a siamese neural network, implemented in Pytorch.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs. The
    'head' model computes the cosine similarity between the embeddings.
    Mimics PyTorch nn.Module API.
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
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train_model(test_generator, test_generator, optimizer, epochs=50)
    """

    def __init__(self,
                 spectrum_binner: SpectrumBinner,
                 base_dims: Tuple[int, ...] = (600, 500, 500),
                 embedding_dim: int = 400,
                 dropout_rate: float = 0.5,
                 dropout_in_first_layer: bool = False,
                 l1_reg: float = 1e-6,
                 l2_reg: float = 1e-6,
                 pytorch_model: nn.Module = None,
                 additional_input=0):
        super(SiameseModel, self).__init__()
        # pylint: disable=too-many-arguments
        assert spectrum_binner.known_bins is not None, \
            "spectrum_binner does not contain known bins (run .fit_transform() on training data first!)"
        self.spectrum_binner = spectrum_binner
        self.input_dim = len(spectrum_binner.known_bins)
        self.additional_input = additional_input

        if pytorch_model is None:
            # Create base model
            self.base = self.get_base_model(input_dim=self.input_dim,
                                            base_dims=base_dims,
                                            embedding_dim=embedding_dim,
                                            dropout_rate=dropout_rate,
                                            dropout_in_first_layer=dropout_in_first_layer,
                                            l1_reg=l1_reg,
                                            l2_reg=l2_reg,
                                            additional_input=additional_input)
            # Create head model
            self.head = self._get_head_model(input_dim=self.input_dim,
                                             additional_input=additional_input,
                                             base_model=self.base)
        else:
            self._construct_from_pytorch_model(pytorch_model)

    def forward(self, input_a, input_b, input_a_2=None, input_b_2=None):
        if self.additional_input > 0:
            embedding_a = self.base(input_a, input_a_2)
            embedding_b = self.base(input_b, input_b_2)
        else:
            embedding_a = self.base(input_a)
            embedding_b = self.base(input_b)

        cosine_similarity = self.head(embedding_a, embedding_b)
        return cosine_similarity

    def save(self, filename: Union[str, Path]):
        """
        Save model to file.
        Parameters
        ----------
        filename
            Filename to specify where to store the model.
        """
        torch.save(self.state_dict(), filename)
        with h5py.File(filename, mode='a') as f:
            f.attrs['spectrum_binner'] = self.spectrum_binner.to_json()
            f.attrs['additional_input'] = self.additional_input

    @staticmethod
    def get_base_model(input_dim: int,
                       base_dims: Tuple[int, ...] = (600, 500, 500),
                       embedding_dim: int = 400,
                       dropout_rate: float = 0.25,
                       dropout_in_first_layer: bool = False,
                       l1_reg: float = 1e-6,
                       l2_reg: float = 1e-6,
                       additional_input=0) -> nn.Module:
        """Create base model for Siamese network.
        Parameters
        ----------
        input_dim : int
            Dimension of the input vectors.
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        dropout_rate
            Dropout rate to be used in the base model
        dropout_in_first_layer
            Set to True if dropout should be part of first dense layer as well. Default is False.
        l1_reg
            L1 regularization rate. Default is 1e-6.
        l2_reg
            L2 regularization rate. Default is 1e-6.
        additional_input
            Default is 0, shape of additional inputs
        """
        # pylint: disable=too-many-arguments, disable=too-many-locals

        layers = []
        dropout_starting_layer = 0 if dropout_in_first_layer else 1

        if additional_input > 0:
            layers.append(nn.Linear(input_dim + additional_input, base_dims[0]))
        else:
            layers.append(nn.Linear(input_dim, base_dims[0]))

        for i, dim in enumerate(base_dims):
            layers.append(nn.ReLU())
            if i >= dropout_starting_layer:
                layers.append(nn.Dropout(dropout_rate))
            if i < len(base_dims) - 1:
                layers.append(nn.Linear(dim, base_dims[i + 1]))

        layers.append(nn.Linear(base_dims[-1], embedding_dim))
        layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _get_head_model(self, input_dim: int,
                        additional_input: int,
                        base_model: nn.Module):
        class CosineSimilarity(nn.Module):
            def forward(self, x1, x2):
                return nn.functional.cosine_similarity(x1, x2, dim=-1, eps=1e-8)

        return CosineSimilarity()

    def _construct_from_pytorch_model(self, pytorch_model):
        def valid_pytorch_model(given_model):
            assert isinstance(given_model, nn.Module), "Expected valid PyTorch model as input."
            assert len(list(given_model.children())) > 0, "Expected more layers"

        valid_pytorch_model(pytorch_model)
        self.base = list(pytorch_model.children())[0]
        self.head = list(pytorch_model.children())[1]

    def train_model(self, train_generator, validation_generator, optimizer, epochs=50):
        self.train()
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_generator, 0):
                # Get the inputs
                inputs_a, inputs_b, labels = data
                if self.additional_input > 0:
                    inputs_a, inputs_a_2 = inputs_a
                    inputs_b, inputs_b_2 = inputs_b

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                if self.additional_input > 0:
                    embeddings_a = self.base(inputs_a, inputs_a_2)
                    embeddings_b = self.base(inputs_b, inputs_b_2)
                else:
                    embeddings_a = self.base(inputs_a)
                    embeddings_b = self.base(inputs_b)

                outputs = self.head(embeddings_a, embeddings_b)

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()

                # Print statistics
                running_loss += loss.item()

            # Compute validation loss
            validation_loss = 0.0
            with torch.no_grad():
                for data in validation_generator:
                    inputs_a, inputs_b, labels = data

                    if self.additional_input > 0:
                        inputs_a, inputs_a_2 = inputs_a
                        inputs_b, inputs_b_2 = inputs_b

                    if self.additional_input > 0:
                        embeddings_a = self.base(inputs_a, inputs_a_2)
                        embeddings_b = self.base(inputs_b, inputs_b_2)
                    else:
                        embeddings_a = self.base(inputs_a)
                        embeddings_b = self.base(inputs_b)

                    outputs = self.head(embeddings_a, embeddings_b)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            # Print epoch summary
            print(f'Epoch {epoch + 1}, Loss: {running_loss / i}, Validation Loss: {validation_loss / len(validation_generator)}')

    def load(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def evaluate(self, test_generator):
        self.eval()
        mse_loss = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for data in test_generator:
                inputs_a, inputs_b, labels = data

                if self.additional_input > 0:
                    inputs_a, inputs_a_2 = inputs_a
                    inputs_b, inputs_b_2 = inputs_b

                if self.additional_input > 0:
                    embeddings_a = self.base(inputs_a, inputs_a_2)
                    embeddings_b = self.base(inputs_b, inputs_b_2)
                else:
                    embeddings_a = self.base(inputs_a)
                    embeddings_b = self.base(inputs_b)

                outputs = self.head(embeddings_a, embeddings_b)
                mse_loss += criterion(outputs, labels).item()

        return mse_loss / len(test_generator)
