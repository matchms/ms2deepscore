from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from matchms.Spectrum import Spectrum
from torch import nn, optim
from ms2deepscore.__version__ import __version__
from ms2deepscore.models.helper_functions import initialize_device
from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator
from ms2deepscore.train_new_model.data_generators import \
    DataGeneratorEmbeddingEvaluation


class EmbeddingEvaluationModel(nn.Module):
    """
    Model to predict the degree of certainty for an MS2DeepScore embedding.

    Attributes
    ----------
    inception_block (InceptionBlock):
        The inception block module used for extracting features.
    global_avg_pool (nn.AdaptiveAvgPool1d):
        Global average pooling layer to reduce feature dimensions.
    fc (nn.Linear):
        Fully connected layer for classification.

    Parameters
    ----------
    input_channels (int):
        Number of input channels (features) in the time series data.
    output_channels (int):
        Number of output classes for classification.
    num_filters (int, optional):
        Number of filters used in convolutional layers. Defaults to 32.
    """
    def __init__(self,
                 settings: SettingsEmbeddingEvaluator,
                 ):
        self.settings = settings
        super().__init__()
        self.inception_block = InceptionBlock(input_channels=1,
                                              num_filters=settings.evaluator_num_filters,
                                              depth=settings.evaluator_depth,
                                              kernel_size=settings.evaluator_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        output_channels = 1
        self.fc = nn.Linear(settings.evaluator_num_filters * 4, output_channels)

    def forward(self, x):
        x = self.inception_block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

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
            'model_params': self.settings.__dict__,
            'model_state_dict': self.state_dict(),
            'version': __version__
        }
        torch.save(settings_dict, filepath)

    def train_evaluator(self,
                        training_spectra: List[Spectrum],
                        ms2ds_model,
                        validation_spectra: List[Spectrum] = None):
        """Train a evaluator model with given parameters.
        """
        data_generator = DataGeneratorEmbeddingEvaluation(spectrums=training_spectra,
                                                          ms2ds_model=ms2ds_model,
                                                          settings=self.settings,
                                                          device="cpu",)
        if validation_spectra is not None:
            val_generator = DataGeneratorEmbeddingEvaluation(spectrums=validation_spectra,
                                                             ms2ds_model=ms2ds_model,
                                                             settings=self.settings,
                                                             device="cpu",)
        else:
            val_generator = None

        device = initialize_device()
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.settings.learning_rate)

        iteration_losses = []
        batch_count = 0  # often we have MANY spectra, so classical epochs are too big --> count batches instead
        for epoch in range(self.settings.num_epochs):
            for i, x in enumerate(data_generator):
                tanimoto_scores, ms2ds_scores, embeddings = x

                for i in range(data_generator.batch_size//self.settings.mini_batch_size):
                    low = i * self.settings.mini_batch_size
                    high = low + self.settings.mini_batch_size

                    optimizer.zero_grad()

                    mse_per_embedding = ((tanimoto_scores[low: high, :] -  ms2ds_scores[low: high, :]) ** 2).mean(axis=1)
                    mse_per_embedding = mse_per_embedding.reshape(-1, 1).clone().detach()

                    outputs = self(embeddings[low: high].reshape(-1, 1, embeddings.shape[-1]).to(device))

                    # Calculate loss
                    loss = criterion(outputs.to(device), mse_per_embedding.to(device, dtype=torch.float32))
                    iteration_losses.append(float(loss))

                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()

                batch_count += 1
                if batch_count % self.settings.batches_per_iteration == 0:
                    print(f">>> Batch: {batch_count} ({batch_count * data_generator.batch_size} spectra, epoch: {epoch + 1})")
                    print(f">>> Training loss: {np.mean(iteration_losses):.6f}")
                    iteration_losses = []
                    if val_generator is not None:
                        with torch.no_grad():
                            self.eval()
                            val_losses = []
                            for sample in val_generator:
                                tanimoto_scores, ms2ds_scores, embeddings = sample
                                outputs = self(embeddings.reshape(-1, 1, embeddings.shape[-1]).to(device))

                                mse_per_embedding = ((tanimoto_scores - ms2ds_scores) ** 2).mean(axis=1)
                                mse_per_embedding = mse_per_embedding.reshape(-1, 1).clone().detach()

                                loss = criterion(outputs.to(device), mse_per_embedding.to(device, dtype=torch.float32))
                                val_losses.append(float(loss))
                            print(f">>> Val_loss: {np.mean(val_losses):.6f}")

                        self.train()

    def compute_embedding_evaluations(self,
                                      embeddings: np.ndarray,
                                      device: str = None,
                                     ):
        """Compute the predicted evaluations of all embeddings.
        """
        embedding_dim = embeddings.shape[1]
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        evaluations = self(torch.tensor(embeddings).reshape(-1, 1, embedding_dim).to(device, dtype=torch.float32))
        return evaluations.cpu().detach().numpy()


class InceptionModule(nn.Module):
    """
    Inception module with bottleneck and convolutional layers.

    Parameters
    ----------
    input_channels (int):
        Number of channels in the input tensor.
    num_filters (int):
        Number of filters in the convolutional layers.
    kernel_size (int, optional):
        Base kernel size for convolutional layers. Defaults to 40.
    use_bottleneck (bool, optional):
        Whether to use a bottleneck layer. Defaults to True.
    """

    def __init__(self, input_channels: int,
                 num_filters: int,
                 kernel_size: int = 40,
                 use_bottleneck: bool = True):
        super().__init__()

        # Create 3 different kernel sizes. Adjust to ensure they are odd for symmetric padding
        kernel_sizes = [max(kernel_size // (2 ** i), 3) for i in range(3)]
        kernel_sizes = [k - (k % 2 == 0) for k in kernel_sizes]

        # Bottleneck layer is only used if input_channels > 1
        use_bottleneck = use_bottleneck if input_channels > 1 else False
        self.bottleneck = nn.Conv1d(input_channels, num_filters, 1, bias=False) if use_bottleneck else nn.Identity()

        # Prepare convolutional layers with adjusted kernel sizes
        conv_input_channels = num_filters if use_bottleneck else input_channels
        self.convs = nn.ModuleList([nn.Conv1d(conv_input_channels, num_filters, k,
                                              padding="same") for k in kernel_sizes])

        # MaxPooling followed by a convolution
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(input_channels, num_filters, 1, bias=False)
        )

        # Batch normalization and activation function
        self.bn = nn.BatchNorm1d(num_filters * 4)

    def forward(self, x):
        bottleneck_output = self.bottleneck(x)
        conv_outputs = [conv(bottleneck_output) for conv in self.convs]
        pooled_output = self.maxconvpool(x)
        concatenated_output = torch.cat(conv_outputs + [pooled_output], dim=1)
        return F.relu(self.bn(concatenated_output))


class InceptionBlock(nn.Module):
    """
    Inception block consisting of multiple Inception modules.

    Parameters
    ----------
    input_channels (int):
        Number of input channels.
    num_filters (int, optional):
        Number of filters for each Inception module. Defaults to 32.
    use_residual (bool, optional):
        Whether to use residual connections. Defaults to True.
    depth (int, optional):
        Number of Inception modules to stack. Defaults to 6.
    """

    def __init__(self,
                 input_channels: int,
                 num_filters: int = 32,
                 use_residual: bool = True,
                 depth: int = 6, **kwargs):
        super().__init__()
        self.use_residual = use_residual
        self.depth = depth
        self.inception_modules = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for d in range(depth):
            module_input_channels = input_channels if d == 0 else num_filters * 4
            self.inception_modules.append(InceptionModule(module_input_channels, num_filters, **kwargs))

            if use_residual and d % 3 == 2:
                shortcut_input_channels = input_channels if d == 2 else num_filters * 4
                if shortcut_input_channels == num_filters * 4:
                    shortcut = nn.BatchNorm1d(shortcut_input_channels)
                else:
                    shortcut = nn.Conv1d(shortcut_input_channels, num_filters * 4, 1, padding="same", bias=False)
                self.shortcuts.append(shortcut)

    def forward(self, x):
        residual = x
        for d in range(self.depth):
            x = self.inception_modules[d](x)
            if self.use_residual and d % 3 == 2:
                shortcut_output = self.shortcuts[d // 3](residual)
                x = x + shortcut_output
                x = F.relu(x)
                residual = x
        return x