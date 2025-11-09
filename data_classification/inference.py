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


def get_dataset_from_kaggle_classification(target_total=5000):
    """
    Charge un CSV de musique, garde les 6 genres les plus fréquents,
    et crée un dataset équilibré d'environ target_total samples.
    
    Args:
        csv_path (str): chemin vers le CSV
        target_total (int): nombre total d'exemples à générer (~5000)
    
    Returns:
        pd.DataFrame: dataset équilibré avec 6 genres
    """
    #df = pd.read_csv("/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/SpotifyFeatures.csv")
    df = pd.read_csv('/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/SpotifyFeatures.csv')
    # Colonnes numériques utiles
    features_numeric = [
        'acousticness', 'danceability', 'duration_ms', 'energy',
        'instrumentalness', 'liveness', 'loudness', 'speechiness', 
        'tempo', 'valence'
    ]

    # Supprimer les lignes avec NaN
    df = df.dropna(subset=['genre'] + features_numeric)
    df = df[features_numeric + ['genre']]

    # Top 6 genres
    top10_genres = df['genre'].value_counts().nlargest(10).index.tolist()

    print("Genres retenus :", top10_genres)

    # Nombre d'exemples par genre
    per_genre = target_total // len(top10_genres)
    print(f"Échantillons par genre : {per_genre} (total ≈ {per_genre*len(top10_genres)})")

    # Créer le dataset équilibré
    df_balanced = (
        df[df['genre'].isin(top10_genres)]
        .groupby('genre')
        .sample(n=per_genre, random_state=42)
        .sample(frac=1, random_state=42)  # shuffle
        .reset_index(drop=True)
    )

    print("Dataset final :", df_balanced.shape)
    return df_balanced
def get_genre_mapping(df):
    genre_mapping = {genre: idx for idx, genre in enumerate(df['genre'].unique())}
    return genre_mapping
# === Définition du MLP identique à celui entraîné ===
class MLP_Net(nn.Module):
    def __init__(self, x_means, x_devs, n_classes=10):
        super().__init__()
        self.x_means = x_means
        self.x_devs = x_devs
        self.linear1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = (x - self.x_means) / self.x_devs
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
class DL_Net(nn.Module):
    
    def __init__(self, x_means, x_deviations, n_classes=10):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1   = nn.Linear(10, 64)
        self.bn1       = nn.BatchNorm1d(64)
        self.act1      = nn.ReLU()
        self.dropout1  = nn.Dropout(0.25)
        
        self.linear2   = nn.Linear(64, 32)
        self.bn2       = nn.BatchNorm1d(32)
        self.act2      = nn.ReLU()
        self.dropout2  = nn.Dropout(0.25)

        self.linear3   = nn.Linear(32, n_classes)  # Pas de Softmax ici (CrossEntropyLoss le fait)

    def forward(self, x):
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
# === Fonction d'inférence ===
def predict_genre(sample_features, model_path, x_means, x_devs, class_labels, n_classes=10):
    model = MLP_Net(x_means, x_devs, n_classes=n_classes)
    #model = DL_Net(x_means, x_devs, n_classes=n_classes)
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

# === Exemple concret ===
# x_means et x_devs doivent être calculés sur ton train set
# Ici, je suppose que tu as déjà ces tensors depuis l'entraînement
# x_means = X_train_tr.mean(0, keepdim=True)
# x_devs  = X_train_tr.std(0, keepdim=True) + 1e-4

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
    # Sample 1 : plutôt énergique et dansant → Pop / Electronic
    np.array([0.12, 0.85, 210000, 0.0, 0.45, 0.3, 0.12, 0.55, 120.0, 0.65], dtype=np.float32),

    # Sample 2 : calme et acoustique → Folk / Indie
    np.array([0.80, 0.40, 180000, 0.25, 0.00, 0.10, -12.0, 0.03, 90.0, 0.20], dtype=np.float32),

    # Sample 3 : vocal et rythmique → Hip-Hop / Comedy
    np.array([0.05, 0.75, 210000, 0.70, 0.00, 0.20, -6.5, 0.20, 100.0, 0.60], dtype=np.float32),

    # Sample 4 : orchestral / instrumental → Soundtrack / Jazz
    np.array([0.15, 0.35, 240000, 0.50, 0.60, 0.30, -10.0, 0.02, 70.0, 0.30], dtype=np.float32),
]

model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/model_classification_MLP_musics_spotify.pt"
#model_path = '/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/model_classification_dl_musics_spotify.pt'
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
# Exemple d'utilisation avec la fonction predict_genre
for i, sample in enumerate(samples_test, 1):
    genre_pred, probs = predict_genre(sample, model_path, x_means, x_deviations, index_to_genre)
    print(f"Sample {i}: 🎵 Genre prédit → {genre_pred}")
    print(f"Probabilités → {probs}\n")