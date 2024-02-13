import torch
import torch.nn.functional as F
from torch import nn


class EmbeddingEvaluationModel(nn.Module):
    """
    Model to predict the degree of certainty for an MS2DeepScore embedding.
    """
    def __init__(self,
                 settings: SettingsMS2Deepscore,
                 ):
        super().__init__()
        self.model_settings = settings
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128, 32) # Adjust the input features to match the flattened output
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class EmbeddingEvaluationModelTrainer(nn.Module):
    """
    Model to predict the degree of certainty for an MS2DeepScore embedding.
    """
    def __init__(self,
                 settings: SettingsMS2Deepscore,
                 ):
        super().__init__()
        self.model_settings = settings
        self.evaluator = EmbeddingEvaluationModel(settings)

    def forward(self, embedding_1, embedding_2):
        # Asses the embedding quality
        eval_1 = self.evaluator(embedding_1)
        eval_2 = self.evaluator(embedding_2)

        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(embedding_1, embedding_2)
        return cos_sim, eval_1, eval_2


def embedding_evaluation_loss(prediction, target, eval_1, eval_2, pretty_bad=0.2):
    embedding_certainty = (eval_1 + eval_2) / 2
    error = (target - prediction)
    confident_loss = embedding_certainty * error ** 2
    uncertain_fallback = (1 - embedding_certainty) * pretty_bad ** 2
    return torch.mean(confident_loss) + torch.mean(uncertain_fallback)
