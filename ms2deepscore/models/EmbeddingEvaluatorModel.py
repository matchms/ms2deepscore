import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.module import Module
from ms2deepscore.__version__ import __version__
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.models.helper_functions import initialize_device


class EmbeddingEvaluationModel(nn.Module):
    """
    Model to predict the degree of certainty for an MS2DeepScore embedding.
    """
    def __init__(self,
                 settings: SettingsMS2Deepscore,
                 ):
        super().__init__()
        self.settings = settings
        self.inception_block = InceptionBlock(1,
                                              self.settings.evaluator_num_filters,
                                              depth=self.settings.evaluator_depth,
                                              kernel_size=self.settings.evaluator_kernel_size,
                                              )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.settings.evaluator_num_filters * 4, 1)        

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


def train_evaluator(evaluator_model,
          data_generator,
          mini_batch_size,
          batches_per_iteration: int,
          learning_rate: float,
          num_epochs,
          val_generator = None):
    """Train a evaluator model with given parameters.

    Parameters
    ----------
    evaluator_model
        The deep learning model to train.
    data_generator
        An iterator for training data batches.
    mini_batch_size
        Defines the actual trainig batch size after which the model weights are optimized.
    learning_rate
        Learning rate for the optimizer.
    val_generator (iterator, optional)
        An iterator for validation data batches.
    """
    # pylint: disable=too-many-arguments, too-many-locals
    device = initialize_device()
    evaluator_model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(evaluator_model.parameters(), lr=learning_rate)


    iteration_losses = []
    batch_count = 0  # often we have MANY spectra, so classical epochs are too big --> count batches instead
    for epoch in range(num_epochs):
        #for i, x in tqdm(enumerate(data_generator)):
        for i, x in enumerate(data_generator):
            tanimoto_scores, ms2ds_scores, embeddings = x

            for i in range(data_generator.batch_size//mini_batch_size):
                low = i * mini_batch_size
                high = low + mini_batch_size

                optimizer.zero_grad()
    
                #rmse_per_embedding = ((remove_diagonal(tanimoto_scores) -  remove_diagonal(ms2ds_scores)) ** 2).mean(axis=1) ** 0.5
                mse_per_embedding = ((tanimoto_scores[low: high, :] -  ms2ds_scores[low: high, :]) ** 2).mean(axis=1)
                mse_per_embedding = mse_per_embedding.reshape(-1, 1).clone().detach()
    
                outputs = evaluator_model(embeddings[low: high].reshape(-1, 1, embeddings.shape[-1]).to(device))
                # Calculate loss
                loss = criterion(outputs.to(device), mse_per_embedding.to(device, dtype=torch.float32))
                iteration_losses.append(float(loss))
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
            batch_count += 1
            if batch_count % batches_per_iteration == 0:
                print(f">>> Batch: {batch_count} ({batch_count * data_generator.batch_size} spectra, epoch: {epoch + 1})")
                print(f">>> Training loss: {np.mean(iteration_losses):.6f}")
                iteration_losses = []
                if val_generator is not None:
                    with torch.no_grad():
                        evaluator_model.eval()
                        val_losses = []
                        for sample in val_generator:
                            tanimoto_scores, ms2ds_scores, embeddings = sample
                            outputs = evaluator_model(embeddings.reshape(-1, 1, embeddings.shape[-1]).to(device))

                            mse_per_embedding = ((tanimoto_scores - ms2ds_scores) ** 2).mean(axis=1)
                            mse_per_embedding = mse_per_embedding.reshape(-1, 1).clone().detach()
                            
                            loss = criterion(outputs.to(device), mse_per_embedding.to(device, dtype=torch.float32))
                            val_losses.append(float(loss))
                        print(f">>> Val_loss: {np.mean(val_losses):.6f}")

                    evaluator_model.train()


### Below is code for a InceptionTime sequence classification model
        

class InceptionTime(Module):
    """
    Default Inception Time model, typically used for time series or sequence classification.
    """
    def __init__(self, input_channels, output_channels, num_filters=32, **kwargs):
        super().__init__()
        self.inception_block = InceptionBlock(input_channels, num_filters, **kwargs)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * 4, output_channels)

    def forward(self, x):
        x = self.inception_block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class InceptionModule(Module):
    def __init__(self, input_channels, num_filters, kernel_size=40, use_bottleneck=True):
        super().__init__()
        # Adjust kernel sizes to ensure they are odd for symmetric padding
        kernel_sizes = [max(kernel_size // (2 ** i), 3) for i in range(3)]
        kernel_sizes = [k - (k % 2 == 0) for k in kernel_sizes]

        # Use bottleneck layer only if input_channels > 1
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


class InceptionBlock(Module):
    def __init__(self, input_channels, num_filters=32, use_residual=True, depth=6, **kwargs):
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


class LinearModel:
    def __init__(self, degree = 2):
        self.degree = degree
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=self.degree)
    
    def fit(self, X, y):
        X_transformed = self.poly.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.poly.transform(X)
        return self.model.predict(X_transformed)

    def save(self, filepath):
        # pylint: disable=protected-access
        # Extract the model's parameters
        model_params = {
            "coef": self.model.coef_.tolist(),  # Convert numpy array to list for JSON serialization
            "intercept": self.model.intercept_.tolist() if hasattr(self.model.intercept_, "tolist") else self.model.intercept_,
            "degree": self.degree,
            "min_degree": self.poly._min_degree,
            "max_degree": self.poly._max_degree,
            "n_output_features_": self.poly.n_output_features_,
            "_n_out_full": self.poly._n_out_full,
        }

        # Export to JSON
        with open(filepath, 'w', encoding="utf-8") as f:
            json.dump(model_params, f)


def load_linear_model(filepath):
    """Load a LinearModel from json.
    """
    # pylint: disable=protected-access
    with open(filepath, "r", encoding="utf-8") as f:
        model_params = json.load(f)

    loaded_model = LinearModel(model_params["degree"])
    loaded_model.model.coef_ = np.array(model_params['coef'])
    loaded_model.model.intercept_ = np.array(model_params['intercept'])
    loaded_model.poly._min_degree = model_params["min_degree"]
    loaded_model.poly._max_degree = model_params["max_degree"]
    loaded_model.poly._n_out_full = model_params["_n_out_full"]
    loaded_model.poly.n_output_features_= model_params["n_output_features_"]
    return loaded_model


def compute_embedding_evaluations(embedding_evaluator: EmbeddingEvaluationModel,
                                  embeddings: np.ndarray,
                                  device: str = None,
                                 ):
    """Compute the predicted evaluations of all embeddings.
    """
    embedding_dim = embeddings.shape[1]
    num_embeddings = embeddings.shape[0]
    evaluations = np.zeros((num_embeddings, ))
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_evaluator.to(device)
    evaluations = embedding_evaluator(torch.tensor(embeddings).reshape(-1, 1, embedding_dim).to(device, dtype=torch.float32))
    return evaluations.cpu().detach().numpy()


def compute_error_predictions(
        embedding_evaluations_1,
        embedding_evaluations_2,
        linear_model):
    n_samples_1 = embedding_evaluations_1.shape[0]
    n_samples_2 = embedding_evaluations_2.shape[0]
    predictions = np.zeros((n_samples_1, n_samples_2))
    
    for i in range(n_samples_1):
        X_pair = np.hstack([np.tile(embedding_evaluations_1[i], n_samples_2).reshape(-1, 1), embedding_evaluations_2])
        
        # Predict using the linear model
        prediction = linear_model.predict(X_pair)
        
        # Store the prediction
        predictions[i, :] = prediction

    return predictions
