import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseSpectralModel(nn.Module):
    """
    Class for training and evaluating a siamese neural network, implemented in PyTorch.
    It consists of a dense 'base' network that produces an embedding for each of the 2 inputs.
    This head model computes the cosine similarity between the embeddings.
    """
    def __init__(self,
                 min_mz=0,
                 max_mz=1000,
                 mz_bin_width=0.01,
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
        min_mz
            Lower bound for m/z values to consider.
        max_mz
            Upper bound for m/z values to consider.
        mz_bin_width
            Bin width for m/z sampling.
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
        l1_reg
            L1 regularization rate. Default is 1e-6.
        l2_reg
            L2 regularization rate. Default is 1e-6.
        keras_model
            When provided, this keras model will be used to construct the SiameseModel instance.
            Default is None.
        """
        super(SiameseSpectralModel, self).__init__()
        self.model_parameters = {
            "min_mz": min_mz,
            "max_mz": max_mz,
            "mz_bin_width": mz_bin_width,
            "base_dims": base_dims,
            "embedding_dim": embedding_dim,
            "train_binning_layer": train_binning_layer,
            "group_size": group_size,
            "output_per_group": output_per_group,
            "dropout_rate": dropout_rate,
            #TODO: add ms2deepscore version
        }
        self.encoder = SpectralEncoder(**self.model_parameters)

    def forward(self, x1, x2):
        # Pass both inputs through the same encoder
        encoded_x1 = self.encoder(x1)
        encoded_x2 = self.encoder(x2)

        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(encoded_x1, encoded_x2)
        return cos_sim

class BinnedSpectraLayer(nn.Module):
    def __init__(self, min_mz, max_mz, mz_bin_width):
        super(BinnedSpectraLayer, self).__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.mz_bin_width = mz_bin_width
        self.num_bins = int((max_mz - min_mz) / mz_bin_width)

    def forward(self, spectra):
        # Assuming spectra is a list of matchms Spectrum objects (with 'peaks.mz' and 'peaks.intensities' attributes)
        binned_spectra = torch.zeros((len(spectra), self.num_bins))

        for i, spectrum in enumerate(spectra):
            for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
                if self.min_mz <= mz < self.max_mz:
                    bin_index = int((mz - self.min_mz) / self.mz_bin_width)
                    binned_spectra[i, bin_index] += intensity

        return binned_spectra

class PeakBinner(nn.Module):
    def __init__(self, input_size, group_size, output_per_group):
        super(PeakBinner, self).__init__()
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
        return torch.cat(outputs, dim=1)

    def output_size(self):
        return self.groups * self.output_per_group
        
class SpectralEncoder(nn.Module):
    def __init__(self, min_mz: float, max_mz: float, mz_bin_width: float,
                 base_dims, embedding_dim, dropout_rate,
                 train_binning_layer: bool, group_size: int, output_per_group: int):
        super(SpectralEncoder, self).__init__()
        self.binning_layer = BinnedSpectraLayer(min_mz, max_mz, mz_bin_width)
        self.train_binning_layer = train_binning_layer

        # First dense layer (no dropout!)
        self.dense_layers = []
        if self.train_binning_layer:
            self.peak_binner = PeakBinner(self.binning_layer.num_bins, group_size, output_per_group)
            self.dense_layers.append(nn.Linear(self.peak_binner.output_size(), base_dims[0]))
        else:
            self.dense_layers.append(nn.Linear(self.binning_layer.num_bins, base_dims[0]))
        input_dim = base_dims[0]

        # Create additional dense layers
        for output_dim in base_dims[1:]:
            self.dense_layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        self.embedding_layer = nn.Linear(base_dims[-1], embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, spectra):
        binned_spectra = self.binning_layer(spectra)
        if self.train_binning_layer:
            x = self.peak_binner(binned_spectra)
            x = F.relu(self.dense_layers[0](x))
        else:
            x = F.relu(self.dense_layers[0](binned_spectra))

        for layer in self.dense_layers[1:]:
            x = F.relu(layer(x))
            x = self.dropout(x)

        x = self.embedding_layer(x)
        return x
