import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


class SiameseSpectralModel(nn.Module):
    """
    Class for training and evaluating a siamese neural network, implemented in PyTorch.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs.
    This head model computes the cosine similarity between the embeddings.
    """
    def __init__(self,
                 peak_inputs: int,
                 additional_inputs: int = 0,
                 base_dims: tuple[int, ...] = (1000, 800, 800),
                 embedding_dim: int = 400,
                 train_binning_layer: bool = True,
                 group_size: int = 30,
                 output_per_group: int = 3,
                 dropout_rate: float = 0.2,
                ):
        """
        Construct SiameseSpectralModel

        Parameters
        ----------
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        train_binning_layer
            Default is True in which case the model contains a first dense multi-group peak binning layer.
        group_size
            When binning layer is used the group_size determins how many input bins are taken into
            one dense micro-network.
        output_per_group
            This sets the number of next layer bins each group_size sized group of inputs shares.
        dropout_rate
            Dropout rate to be used in the base model.
        peak_inputs
            Integer to specify the number of binned peaks in the input spectra.
        additional_inputs
            Integer to specify the number of additional (metadata) input fields.
        pytorch_model
            When provided, this pytorch model will be used to construct the SiameseModel instance.
            Default is None.
        """
        # pylint: disable=too-many-arguments
        super().__init__()
        self.model_parameters = {
            "base_dims": base_dims,
            "embedding_dim": embedding_dim,
            "train_binning_layer": train_binning_layer,
            "group_size": group_size,
            "output_per_group": output_per_group,
            "dropout_rate": dropout_rate,
            "peak_inputs": peak_inputs,
            "additional_inputs": additional_inputs,
            #TODO: add ms2deepscore version
        }
        self.encoder = SpectralEncoder(**self.model_parameters)

    def forward(self, spectra_tensors_1, spectra_tensors_2, metadata_1, metadata_2):
        # Pass both inputs through the same encoder
        encoded_x1 = self.encoder(spectra_tensors_1, metadata_1)
        encoded_x2 = self.encoder(spectra_tensors_2, metadata_2)

        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(encoded_x1, encoded_x2)
        return cos_sim


class PeakBinner(nn.Module):
    def __init__(self, input_size, group_size, output_per_group):
        super().__init__()
        self.group_size = group_size
        self.output_per_group = output_per_group
        self.groups = input_size // group_size

        # Create a ModuleList of linear layers, each mapping group_size inputs to output_per_group outputs
        self.linear_layers = nn.ModuleList([nn.Linear(group_size, output_per_group) for _ in range(self.groups)])

    def forward(self, x):
        # Split the input into groups and apply each linear layer to each group
        outputs = [linear(x[:, i*self.group_size:(i+1)*self.group_size]) for i, linear in enumerate(self.linear_layers)]

        # Make sure all inputs get a connection to the next layer
        i = self.groups - 1
        outputs[-1] = self.linear_layers[-1](x[:, i*self.group_size:(i+1)*self.group_size])

        # Concatenate all outputs
        return F.relu(torch.cat(outputs, dim=1))

    def output_size(self):
        return self.groups * self.output_per_group


class SpectralEncoder(nn.Module):
    def __init__(self,
                 base_dims,
                 embedding_dim,
                 dropout_rate,
                 train_binning_layer: bool, group_size: int, output_per_group: int,
                 peak_inputs: int,
                 additional_inputs: int,
                 ):
        """
        Parameters
        ----------
        base_dims
            Tuple of integers depicting the dimensions of the desired hidden
            layers of the base model
        embedding_dim
            Dimension of the embedding (i.e. the output of the base model)
        train_binning_layer
            Default is True in which case the model contains a first dense multi-group peak binning layer.
        group_size
            When binning layer is used the group_size determins how many input bins are taken into
            one dense micro-network.
        output_per_group
            This sets the number of next layer bins each group_size sized group of inputs shares.
        dropout_rate
            Dropout rate to be used in the base model.
        peak_inputs
            Integer to specify the number of binned peaks in the input spectra.
        additional_inputs
            Integer to specify the number of additional (metadata) input fields.
        """
        # pylint: disable=too-many-arguments
        super().__init__()
        #self.binning_layer = BinnedSpectraLayer(min_mz, max_mz, mz_bin_width, intensity_scaling)
        self.train_binning_layer = train_binning_layer

        # First dense layer (no dropout!)
        self.dense_layers = nn.ModuleList()
        if self.train_binning_layer:
            self.peak_binner = PeakBinner(peak_inputs,
                                          group_size, output_per_group)
            input_size = self.peak_binner.output_size() + additional_inputs
        else:
            input_size = peak_inputs + additional_inputs
        self.dense_layers.append(nn.Linear(input_size, base_dims[0]))
        input_dim = base_dims[0]

        # Create additional dense layers
        for output_dim in base_dims[1:]:
            self.dense_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        self.embedding_layer = nn.Linear(base_dims[-1], embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, spectra_tensors, metadata_tensors):
        if self.train_binning_layer:
            x = self.peak_binner(spectra_tensors)
            x = torch.cat([metadata_tensors, x], dim=1)
        else:
            x = torch.cat([metadata_tensors, spectra_tensors], dim=1)
        x = F.relu(self.dense_layers[0](x))

        for layer in self.dense_layers[1:]:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.embedding_layer(x)
        return x


### Model training

def train(model: torch.nn.Module,
          data_generator,
          num_epochs: int,
          learning_rate: float,
          val_generator=None, 
          early_stopping=True,
          patience: int = 10,
          checkpoint_filename: str = None, 
          lambda_l1: float = 1e-6,
          lambda_l2: float = 1e-6):
    """Train a model with given parameters.

    Parameters
    ----------
    model
        The neural network model to train.
    data_generator
        An iterator for training data batches.
    num_epochs
        Number of epochs for training.
    learning_rate
        Learning rate for the optimizer.
    val_generator (iterator, optional)
        An iterator for validation data batches.
    early_stopping
        Whether to use early stopping.
    patience
        Number of epochs to wait for improvement before stopping.
    checkpoint_filename
        File path to save the model checkpoint.
    lambda_l1
        L1 regularization strength.
    lambda_l2 
        L2 regularization strength.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    device = initialize_device()
    criterion, optimizer = setup_model(model, learning_rate, device)
    model.to(device)

    losses, val_losses, collection_targets = [], [], []
    min_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train(True)
        with tqdm(data_generator, unit="batch", mininterval=0) as training:
            training.set_description(f"Epoch {epoch}")
            batch_losses = []
            for spectra_1, spectra_2, meta_1, meta_2, targets in training:
                # For debugging: keep track of biases
                collection_targets.extend(targets)
                
                optimizer.zero_grad()

                # Forward pass
                outputs = model(spectra_1.to(device), spectra_2.to(device), 
                                meta_1.to(device), meta_2.to(device))

                # Calculate loss
                loss = criterion(outputs, targets.to(device))
                if lambda_l1 > 0 or lambda_l2 > 0:
                    loss += l1_regularization(model, lambda_l1) + l2_regularization(model, lambda_l2)
                batch_losses.append(float(loss))

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print progress
                training.set_postfix(
                    loss=float(loss),
                )

        if val_generator is not None:
            model.eval()
            val_batch_losses = []
            for spectra_1, spectra_2, meta_1, meta_2, targets in val_generator:
                predictions = model(spectra_1.to(device), spectra_2.to(device), 
                                    meta_1.to(device), meta_2.to(device))
                loss = criterion(predictions, targets.to(device))
                val_batch_losses.append(float(loss))
            val_loss = np.mean(val_batch_losses)
            val_losses.append(val_loss)
            if val_loss < min_val_loss:
                if checkpoint_filename:
                    print("Saving checkpoint model.")
                    torch.save(model, checkpoint_filename)
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print("Early stopping!")
                break

        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(batch_losses):.4f}")
        losses.append(np.mean(batch_losses))
        if val_generator is not None:
            print(f"Validation Loss: {val_loss:.4f}")

    return losses, val_losses, collection_targets


def initialize_device():
    """Initialize and return the device for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training will happen on {device}.")
    return device


def setup_model(model, learning_rate, device):
    """
    Set up the model for training.

    Parameters
    ----------
    model
        The model to be set up.
    learning_rate
        Learning rate for the optimizer.
    device
        The device to be used for training.
    """
    model.to(device)
    criterion = mse_away_from_mean  # Alternative for nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


### Helper functions

def l1_regularization(model, lambda_l1):
    """L1 regulatization for first dense layer of model."""
    l1_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=1)
    return lambda_l1 * l1_loss

def l2_regularization(model, lambda_l2):
    """L2 regulatization for first dense layer of model."""
    l2_loss = torch.linalg.vector_norm(next(model.encoder.dense_layers[0].parameters()), ord=2)
    return lambda_l2 * l2_loss

def mse_away_from_mean(output, target):
    """MSE weighted to get higher loss for predictions towards the mean of 0.5.
    
    In addition, we are usually more intereted in the precision for higher scores.
    And, we have often fewer pairs in that regime. This is included by an additional
    linear factor to shift attention to higher scores.
    """
    weighting = torch.exp(-10 * (output - 0.5)**2) + 1
    focus_high_scores = 1 + 0.5 * target
    loss = torch.mean(weighting * focus_high_scores * (output - target)**2)
    return loss
