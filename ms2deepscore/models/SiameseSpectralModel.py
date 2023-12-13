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
                 step=0.01,
                 train_binning_layer: bool = True,
                 group_size: int = 30,
                 output_per_group: int = 3,
                ):
        super(SiameseSpectralModel, self).__init__()
        self.model_parameters = {
            "min_mz": min_mz,
            "max_mz": max_mz,
            "step": step,
            "train_binning_layer": train_binning_layer,
            "group_size": group_size,
            "output_per_group": output_per_group,
            #TODO: add ms2deepscore version
        }
        self.encoder = SpectralEncoder(min_mz, max_mz, step, train_binning_layer, group_size, output_per_group)

    def forward(self, x1, x2):
        # Pass both inputs through the same encoder
        encoded_x1 = self.encoder(x1)
        encoded_x2 = self.encoder(x2)

        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(encoded_x1, encoded_x2)
        return cos_sim

class BinnedSpectraLayer(nn.Module):
    def __init__(self, min_mz, max_mz, step):
        super(BinnedSpectraLayer, self).__init__()
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.step = step
        self.num_bins = int((max_mz - min_mz) / step)

    def forward(self, spectra):
        # Assuming spectra is a list of matchms Spectrum objects (with 'peaks.mz' and 'peaks.intensities' attributes)
        binned_spectra = torch.zeros((len(spectra), self.num_bins))

        for i, spectrum in enumerate(spectra):
            for mz, intensity in zip(spectrum.peaks.mz, spectrum.peaks.intensities):
                if self.min_mz <= mz < self.max_mz:
                    bin_index = int((mz - self.min_mz) / self.step)
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
    def __init__(self, min_mz: float, max_mz: float, step: float,
                 train_binning_layer: bool, group_size: int, output_per_group: int):
        super(SpectralEncoder, self).__init__()
        self.binning_layer = BinnedSpectraLayer(min_mz, max_mz, step)
        self.train_binning_layer = train_binning_layer

        if self.train_binning_layer:
            self.peak_binner = PeakBinner(self.binning_layer.num_bins, 30, 3)
            self.fc1 = nn.Linear(self.peak_binner.output_size(), 1000)
        else:
            self.fc1 = nn.Linear(self.binning_layer.num_bins, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)
        self.output_layer = nn.Linear(1000, 500)

    def forward(self, spectra):
        binned_spectra = self.binning_layer(spectra)
        if self.train_binning_layer:
            x = self.peak_binner(binned_spectra)
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.fc1(binned_spectra))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_layer(x)
        return x
