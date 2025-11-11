import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from mlxtend.plotting import heatmap

np.set_printoptions(precision=4, suppress=True)


def get_dataset_from_kaggle_classification(target_total=5000) -> pd.DataFrame:
    """
    Loads a music CSV, keeps the 10 most frequent genres,
    and creates a balanced dataset with approximately target_total samples.
    
    Args:
        target_total (int): total number of examples to generate (~5000)
    
    Returns:
        pd.DataFrame: balanced dataset with 10 genres
    
    """
    df = pd.read_csv('./data_classification/SpotifyFeatures.csv')
    
    # Useful numeric columns
    features_numeric = [
        'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 
        'tempo', 'valence'
    ]

    # Drop rows with NaN
    df = df.dropna(subset=['genre'] + features_numeric)
    df = df[features_numeric + ['genre']]

    # Top 10 genres
    top10_genres = df['genre'].value_counts().nlargest(10).index.tolist()

    print("Selected genres:", top10_genres)

    # Number of examples per genre
    per_genre = target_total // len(top10_genres)
    print(f"Samples per genre: {per_genre} (total ≈ {per_genre*len(top10_genres)})")

    # Create balanced dataset
    df_balanced = (
        df[df['genre'].isin(top10_genres)]
        .groupby('genre')
        .sample(n=per_genre, random_state=42)
        .sample(frac=1, random_state=42)  # shuffle
        .reset_index(drop=True)
    )

    print("Final dataset:", df_balanced.shape)
    return df_balanced


def get_genre_mapping(df: pd.DataFrame) -> dict:
    """
    Creates a mapping from genre names to numeric class indices.
    
    Args:
        df (pd.DataFrame): Dataset containing the 'genre' column.
    
    Returns:
        dict: Mapping from genre to integer index.
    """
    genre_mapping = {genre: idx for idx, genre in enumerate(df['genre'].unique())}
    return genre_mapping


class MLP_Net(nn.Module):
    """
    Multi-layer Perceptron (MLP) for music genre classification.
    """
    def __init__(self, x_means: torch.Tensor, x_devs: torch.Tensor, n_classes: int = 10):
        super().__init__()
        self.x_means = x_means
        self.x_devs = x_devs
        self.linear1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.x_means) / self.x_devs
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DL_Net(nn.Module):
    """
    Deep neural network with two hidden layers for music genre classification.
    """
    def __init__(self, x_means: torch.Tensor, x_deviations: torch.Tensor, n_classes: int = 10):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations

        self.linear1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.linear2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        self.linear3 = nn.Linear(32, n_classes)  # logits, no softmax here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.x_means) / self.x_deviations

        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        return x  # logits


def predict_genre(
    sample_features: np.ndarray,
    model_path: str,
    x_means: torch.Tensor,
    x_devs: torch.Tensor,
    class_labels: dict,
    n_classes: int = 10
) -> tuple:
    """
    Predicts the genre of a single music sample using a trained PyTorch model.

    Args:
        sample_features (np.ndarray): Feature vector of a single sample.
        model_path (str): Path to the trained model (.pt file).
        x_means (torch.Tensor): Tensor of training feature means.
        x_devs (torch.Tensor): Tensor of training feature standard deviations.
        class_labels (dict): Mapping from class index to genre name.
        n_classes (int): Number of output classes.

    Returns:
        tuple: (predicted genre (str), probabilities (np.ndarray))
    """
    model = MLP_Net(x_means, x_devs, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    x = torch.tensor(sample_features, dtype=torch.float32).unsqueeze(0)
    x = (x - x_means) / x_devs

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        predicted_genre = class_labels[pred_class]

    return predicted_genre, probs.numpy()


# === Example usage ===
index_to_genre = {
    0: 'Folk',
    1: 'Indie',
    2: 'Electronic',
    3: 'Comedy',
    4: 'Children’s Music',
    5: 'Hip-Hop',
    6: 'Jazz',
    7: 'Pop',
    8: 'Soundtrack',
    9: 'Rock'
}

samples_test = [
    np.array([0.12, 0.85, 210000, 0.0, 0.45, 0.3, 0.12, 0.55, 120.0, 0.65], dtype=np.float32),
    np.array([0.80, 0.40, 180000, 0.25, 0.00, 0.10, -12.0, 0.03, 90.0, 0.20], dtype=np.float32),
    np.array([0.05, 0.75, 210000, 0.70, 0.00, 0.20, -6.5, 0.20, 100.0, 0.60], dtype=np.float32),
    np.array([0.15, 0.35, 240000, 0.50, 0.60, 0.30, -10.0, 0.02, 70.0, 0.30], dtype=np.float32),
]

model_path = "./data_classification/model_classification_MLP_musics_spotify.pt"
epsilon = 0.0001

musics = get_dataset_from_kaggle_classification()
genre_mapping = get_genre_mapping(musics)
print("Genre mapping:", genre_mapping)

musics['genre'] = musics['genre'].map(genre_mapping)
musics_np = musics.to_numpy()

X = musics_np[:, :-1].astype(np.float32)
Y = musics_np[:, -1:].astype(np.int64)

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

X_train_tr = torch.from_numpy(X_train)
X_test_tr = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_test_tr = torch.from_numpy(y_test)

x_means = X_train_tr.mean(0, keepdim=True)
x_deviations = X_train_tr.std(0, keepdim=True) + epsilon

for i, sample in enumerate(samples_test, 1):
    genre_pred, probs = predict_genre(sample, model_path, x_means, x_deviations, index_to_genre)
    print(f"Sample {i}: Predicted genre → {genre_pred}")
    print(f"Probabilities → {probs}\n")
