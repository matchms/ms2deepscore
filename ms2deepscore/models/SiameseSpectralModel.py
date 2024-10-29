import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ms2deepscore.__version__ import __version__
from ms2deepscore.models.helper_functions import (initialize_device,
                                                  l1_regularization,
                                                  l2_regularization)
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS, rmse_loss
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra


class SiameseSpectralModel(nn.Module):
    """
    Class for training and evaluating a Siamese neural network, implemented in PyTorch.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs.
    This head model computes the cosine similarity between the embeddings.
    """
    def __init__(self,
                 settings: SettingsMS2Deepscore,
                 ):
        super().__init__()
        self.model_settings = settings
        self.encoder = SpectralEncoder(settings=self.model_settings, peak_inputs=self.model_settings.number_of_bins(),
                                       additional_inputs=len(self.model_settings.additional_metadata))

    def forward(self, spectra_tensors_1, spectra_tensors_2, metadata_1, metadata_2):
        # Pass both inputs through the same encoder
        encoded_x1 = self.encoder(spectra_tensors_1, metadata_1)
        encoded_x2 = self.encoder(spectra_tensors_2, metadata_2)

        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(encoded_x1, encoded_x2)
        return cos_sim

    def save(self, filepath):
        """
        Save the model's parameters and state dictionary to a file.

        Parameters
        ----------
        filepath: str
            The file path where the model will be saved.
        """
        # Ensure the model is in evaluation mode
        self.eval()
        settings_dict = {
            'model_params': self.model_settings.__dict__,
            'model_state_dict': self.state_dict(),
            'version': __version__
        }
        torch.save(settings_dict, filepath)


class PeakBinner(nn.Module):
    """
    This model element is meant to be a "smart binning" element to reduce
    a high number of inputs by using many smaller densely connected units (groups).
    The initial input tensors will thereby be divided into groups of `group_size` inputs
    which are connected to `output_per_group` outputs.
    """
    def __init__(self, input_size, settings: SettingsMS2Deepscore):
        super().__init__()
        self.group_size = settings.train_binning_layer_group_size
        self.step_width = int(settings.train_binning_layer_group_size / 2)
        self.output_per_group = settings.train_binning_layer_output_per_group
        self.groups = 2 * input_size // settings.train_binning_layer_group_size - 1 # overlapping groups

        # Create a ModuleList of linear layers, each mapping group_size inputs to output_per_group outputs
        self.linear_layers = nn.ModuleList([nn.Linear(settings.train_binning_layer_group_size,
                                                      settings.train_binning_layer_output_per_group,
                                                      bias=False) for _ in range(self.groups)])

        # Initialize weights
        for x in self.linear_layers:
            nn.init.uniform_(x.weight, 0.9, 1.1)

    def forward(self, x):
        # Split the input into groups and apply each linear layer to each group
        outputs = [linear(x[:, i*self.step_width :(i+2)*self.step_width ]) for i, linear in enumerate(self.linear_layers)]

        # Make sure all inputs get a connection to the next layer
        i = self.groups - 1
        outputs[-1] = self.linear_layers[-1](x[:, i*self.step_width:(i+2)*self.step_width ])

        # Concatenate all outputs
        return F.relu(torch.cat(outputs, dim=1))

    def output_size(self):
        return self.groups * self.output_per_group


class SpectralEncoder(nn.Module):
    """Encoder part of the Siamese network.

    This model will convert input spectra (spectra_tensors and metadata_tensors) to reduced embeddings.
    """
    def __init__(self,
                 settings: SettingsMS2Deepscore,
                 peak_inputs: int,
                 additional_inputs: int,
                 ):
        """
        Parameters
        ----------
        settings:
            SettingsMS2Deepscore object storing all the settings
        peak_inputs
            Integer to specify the number of binned peaks in the input spectra.
        additional_inputs
            Integer to specify the number of additional (metadata) input fields.
        """
        super().__init__()
        self.train_binning_layer = settings.train_binning_layer

        # First dense layer (no dropout!)
        self.dense_layers = nn.ModuleList()
        if self.train_binning_layer:
            self.peak_binner = PeakBinner(peak_inputs, settings)
            input_size = self.peak_binner.output_size() + additional_inputs
        else:
            input_size = peak_inputs + additional_inputs
        self.dense_layers.append(
            dense_layer(input_size, settings.base_dims[0], settings.activation_function)
        )
        input_dim = settings.base_dims[0]

        # Create additional dense layers
        for output_dim in settings.base_dims[1:]:
            self.dense_layers.append(dense_layer(input_dim, output_dim, settings.activation_function))
            input_dim = output_dim

        self.embedding_layer = dense_layer(settings.base_dims[-1], settings.embedding_dim, "tanh")
        self.dropout = nn.Dropout(settings.dropout_rate)

    def forward(self, spectra_tensors, metadata_tensors):
        if self.train_binning_layer:
            x = self.peak_binner(spectra_tensors)
            x = torch.cat([metadata_tensors, x], dim=1)
        else:
            x = torch.cat([metadata_tensors, spectra_tensors], dim=1)
        x = self.dense_layers[0](x)

        for layer in self.dense_layers[1:]:
            x = layer(x)
            x = self.dropout(x)

        x = self.embedding_layer(x)
        return x


def train(model: SiameseSpectralModel,
          data_generator,
          num_epochs: int,
          learning_rate: float,
          validation_loss_calculator = None,
          early_stopping=True,
          patience: int = 10,
          checkpoint_filename: str = None,
          loss_function="MSE",
          weighting_factor=0,
          monitor_rmse: bool = True,
          collect_all_targets: bool = False,
          lambda_l1: float = 0,
          lambda_l2: float = 0,
          progress_bar: bool = True,
          log_dir: str = "./runs"):
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
    loss_function
        Pass a loss function (e.g. a pytorch default or a custom function).
    weighting_factor
        Default is set to 0, set to value between 0 and 1 to shift attention to higher target scores.
    monitor_rmse
        If True rmse will be monitored turing training.
    collect_all_targets
        If True, all training targets will be collected (e.g. for later statistics).
    lambda_l1
        L1 regularization strength.
    lambda_l2 
        L2 regularization strength.
    """
    device = initialize_device()
    model.to(device)

    if loss_function.lower() not in LOSS_FUNCTIONS:
        raise ValueError(f"Unknown loss function. Must be one of: {LOSS_FUNCTIONS.keys()}")
    criterion = LOSS_FUNCTIONS[loss_function.lower()]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize TensorBoard writer
    if checkpoint_filename:
        writer = SummaryWriter(log_dir=os.path.join(log_dir, checkpoint_filename.split(".")[0]))
    else:
        writer = SummaryWriter(log_dir=log_dir)

    history = {
        "losses": [],
        "val_losses": [],
        "rmse": [],
        "val_rmse": [],
        "collection_targets": [],
        }
    min_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train(True)
        with tqdm(data_generator, unit="batch", mininterval=0, disable=(not progress_bar)) as training:
            training.set_description(f"Epoch {epoch}")
            batch_losses = []
            batch_rmse = []
            for spectra_1, spectra_2, meta_1, meta_2, targets in training:
                if collect_all_targets:
                    history["collection_targets"].extend(targets)
                
                optimizer.zero_grad()

                # Forward pass
                outputs = model(spectra_1.to(device), spectra_2.to(device), 
                                meta_1.to(device), meta_2.to(device))

                # Calculate loss
                loss = criterion(outputs, targets.to(device), weighting_factor=weighting_factor)
                if lambda_l1 > 0 or lambda_l2 > 0:
                    loss += l1_regularization(model, lambda_l1) + l2_regularization(model, lambda_l2)
                batch_losses.append(float(loss))

                if monitor_rmse:
                    batch_rmse.append(rmse_loss(outputs, targets.to(device)).cpu().detach().numpy())

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print progress
                training.set_postfix(
                    loss=float(loss),
                    rmse=np.mean(batch_rmse),
                )
        epoch_loss = np.mean(batch_losses)
        epoch_rmse = np.mean(batch_rmse) if monitor_rmse else None

        history["losses"].append(epoch_loss)
        history["rmse"].append(epoch_rmse)

        writer.add_scalar('Loss/train', epoch_loss, epoch)

        if validation_loss_calculator is not None:
            val_losses, _  = validation_loss_calculator.compute_binned_validation_loss(
                model,
                loss_types=(loss_function, "rmse")
                )
            val_loss = val_losses[loss_function]
            val_rmse = val_losses["rmse"]
            history["val_losses"].append(val_loss)
            history["val_rmse"].append(val_rmse)

            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('RMSE/validation', val_rmse, epoch)

            if val_loss < min_val_loss:
                if checkpoint_filename:
                    print("Saving checkpoint model.")
                    model.save(checkpoint_filename)
                epochs_no_improve = 0
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print("Early stopping!")
                break

        # Print statistics
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(batch_losses):.4f}")
        if validation_loss_calculator is not None:
            print(f"Validation Loss: {val_loss:.4f} (RMSE: {val_losses['rmse']:.4f}).")

    writer.close()
    return history


def dense_layer(input_size, output_size, activation="lrelu"):
    """Combines a densely connected layer and an activation function."""
    activations = nn.ModuleDict([
        ["lrelu", nn.LeakyReLU()],
        ["relu", nn.ReLU()],
        ["sigmoid", nn.Sigmoid()],
        ["tanh", nn.Tanh()]
    ])
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activations[activation]
    )


def compute_embedding_array(model: SiameseSpectralModel,
                            spectrums,
                            datatype="numpy",
                            device=None):
    """Compute the embeddings of all spectra in spectrums.
    """
    if datatype.lower() not in ["numpy", "pytorch"]:
        raise ValueError("datatype can only be 'numpy' or 'pytorch'.")
    if datatype.lower() == "numpy":
        embeddings = np.zeros((len(spectrums), model.model_settings.embedding_dim))
    else:
        embeddings = torch.zeros((len(spectrums), model.model_settings.embedding_dim))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i, spec in tqdm(enumerate(spectrums)):
        X = tensorize_spectra([spec], model.model_settings)
        with torch.no_grad():
            if datatype.lower() == "numpy":
                embeddings[i, :] = model.encoder(X[0].to(device), X[1].to(device)).cpu().detach().numpy()
            else:
                embeddings[i, :] = model.encoder(X[0].to(device), X[1].to(device)).cpu().detach()
    return embeddings
