import pandas as pd
import numpy as np
import torch
import sklearn
import random
import torch
import onnx
import onnxruntime as ort
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

############################################
# 1) Chargement du dataset
############################################

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
    df = pd.read_csv("/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/SpotifyFeatures.csv")

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
    return {genre: idx for idx, genre in enumerate(df['genre'].unique())}

############################################
# 2) Réseau profond (2 hidden layers)
############################################

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

############################################
# 3) Boucle d'entraînement
############################################

def training_loop(N_Epochs, model, loss_fn, opt, train_dl):
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss   = loss_fn(y_pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 100 == 0:
            print(epoch, "loss=", loss)

############################################
# 4) Fonction métriques
############################################

def print_metrics_function(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix:")
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))

############################################
# 5) MAIN
############################################

def main():
    np.set_printoptions(precision=4, suppress=True)
    batch_size    = 64 # Car modele pas enorme 
    learning_rate = 0.0005## 0.001
    N_Epochs      = 4000
    epsilon = 0.0001

    musics = get_dataset_from_kaggle_classification()

    genre_mapping = get_genre_mapping(musics)
    musics['genre'] = musics['genre'].map(genre_mapping)
    musics_np = musics.to_numpy()
    
    X = musics_np[:, :-1]
    Y = musics_np[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    X_train_tr = torch.from_numpy(X_train)
    X_test_tr  = torch.from_numpy(X_test)
    y_train_tr = torch.from_numpy(y_train)
    y_test_tr  = torch.from_numpy(y_test)

    x_means      = X_train_tr.mean(0, keepdim=True)
    x_deviations = X_train_tr.std(0, keepdim=True) + epsilon

    musics_train_list = [  ( X_train_tr[i],  y_train_tr[i].item()  )  for i in range( X_train.shape[0] ) ]

    musics_test_list  = [  ( X_test_tr[i],   y_test_tr[i].item()   )  for i in range( X_test.shape[0]  ) ]
    train_dl = torch.utils.data.DataLoader(musics_train_list, batch_size=batch_size, shuffle=True)
    all_test_data = X_test.shape[0]
    test_dl  = torch.utils.data.DataLoader(musics_test_list,  batch_size=all_test_data, shuffle=True)
    
    model = DL_Net(x_means, x_deviations)
    opt   = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    training_loop(N_Epochs, model, loss_fn, opt, train_dl)
    torch.save(model.state_dict(), "model_classification_DL_musics_spotify.pt")
    
    
    # Dummy input pour ONNX (la forme doit correspondre à ton modèle)
    # Ici batch_size=1 et 10 features
    dummy_input = torch.randn(1, 10, dtype=torch.float32)

    # Export ONNX
    onnx_model_path = "model_classification_DL_musics_spotify.onnx"
    model.eval()
    torch.onnx.export(
        model,                     # modèle PyTorch
        dummy_input,               # exemple d'entrée
        onnx_model_path,           # chemin de sauvegarde
        input_names=['input'],     # nom de l'entrée
        output_names=['output'],   # nom de la sortie
        dynamic_axes={
            'input': {0: 'batch_size'},   # batch_size dynamique
            'output': {0: 'batch_size'}
        },
        opset_version=18           # version ONNX (récent)
    )

    print(f"✅ Modèle ONNX sauvegardé : {onnx_model_path}")
    
    
    with torch.no_grad():
        for x_real, y_real in test_dl:
            logits = model(x_real)
            preds = torch.argmax(logits, dim=1)
            print_metrics_function(y_real.squeeze().numpy(), preds.numpy())

if __name__ == "__main__":
    main()

'''
Genres retenus : ['Comedy', 'Soundtrack', 'Indie', 'Jazz', 'Pop', 'Electronic', 'Children’s Music', 'Folk', 'Hip-Hop', 'Rock']
Échantillons par genre : 500 (total ≈ 5000)
Dataset final : (5000, 11)
0 loss= tensor(2.1507, grad_fn=<NllLossBackward0>)
100 loss= tensor(1.4537, grad_fn=<NllLossBackward0>)
200 loss= tensor(1.1782, grad_fn=<NllLossBackward0>)
300 loss= tensor(1.2776, grad_fn=<NllLossBackward0>)
400 loss= tensor(1.2968, grad_fn=<NllLossBackward0>)
500 loss= tensor(1.3947, grad_fn=<NllLossBackward0>)
600 loss= tensor(1.0861, grad_fn=<NllLossBackward0>)
700 loss= tensor(1.0494, grad_fn=<NllLossBackward0>)
800 loss= tensor(1.6447, grad_fn=<NllLossBackward0>)
900 loss= tensor(1.2455, grad_fn=<NllLossBackward0>)
1000 loss= tensor(1.3367, grad_fn=<NllLossBackward0>)
1100 loss= tensor(1.1433, grad_fn=<NllLossBackward0>)
1200 loss= tensor(1.6087, grad_fn=<NllLossBackward0>)
1300 loss= tensor(1.2118, grad_fn=<NllLossBackward0>)
1400 loss= tensor(1.2450, grad_fn=<NllLossBackward0>)
1500 loss= tensor(1.4591, grad_fn=<NllLossBackward0>)
1600 loss= tensor(1.3884, grad_fn=<NllLossBackward0>)
1700 loss= tensor(1.1545, grad_fn=<NllLossBackward0>)
1800 loss= tensor(1.7574, grad_fn=<NllLossBackward0>)
1900 loss= tensor(1.2933, grad_fn=<NllLossBackward0>)
2000 loss= tensor(1.4465, grad_fn=<NllLossBackward0>)
2100 loss= tensor(1.2289, grad_fn=<NllLossBackward0>)
2200 loss= tensor(1.2781, grad_fn=<NllLossBackward0>)
2300 loss= tensor(1.2324, grad_fn=<NllLossBackward0>)
2400 loss= tensor(1.2593, grad_fn=<NllLossBackward0>)
2500 loss= tensor(1.3964, grad_fn=<NllLossBackward0>)
2600 loss= tensor(1.1671, grad_fn=<NllLossBackward0>)
2700 loss= tensor(1.5792, grad_fn=<NllLossBackward0>)
2800 loss= tensor(1.0575, grad_fn=<NllLossBackward0>)
2900 loss= tensor(1.3543, grad_fn=<NllLossBackward0>)
3000 loss= tensor(1.3858, grad_fn=<NllLossBackward0>)
3100 loss= tensor(1.1871, grad_fn=<NllLossBackward0>)
3200 loss= tensor(1.1839, grad_fn=<NllLossBackward0>)
3300 loss= tensor(1.3189, grad_fn=<NllLossBackward0>)
3400 loss= tensor(1.2383, grad_fn=<NllLossBackward0>)
3500 loss= tensor(1.3619, grad_fn=<NllLossBackward0>)
3600 loss= tensor(1.2097, grad_fn=<NllLossBackward0>)
3700 loss= tensor(1.3086, grad_fn=<NllLossBackward0>)
3800 loss= tensor(1.2693, grad_fn=<NllLossBackward0>)
3900 loss= tensor(1.2616, grad_fn=<NllLossBackward0>)
Accuracy: 0.45
Confusion Matrix:
[[52  6  4  0 10  2  6  7  3 11]
 [35  6 10  1 15  8  8 15  2  8]
 [ 4  3 57  0  6 11 13  6  6  7]
 [ 2  0  0 86  1  3  2  0  0  0]
 [ 2  7  5  0 33 11  4  6  0 16]
 [ 0  4  6  0  2 54  1 16  0  1]
 [24  3 15  0  3  8 43  3  7  3]
 [10  4  2  0 12 29  1 25  1 15]
 [ 5  1  1  0  2  1  3  1 86  0]
 [32  5  5  0 31  5  2 13  3 12]]
Precision: 0.442
Recall: 0.454
F1-measure: 0.438
'''