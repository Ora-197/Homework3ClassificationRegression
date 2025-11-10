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

########################################
# DATASET
########################################

def get_dataset_from_kaggle_classification(target_total=5000):
    """
    Loads a music CSV, keeps the 10 most frequent genres,
    and creates a balanced dataset with approximately target_total samples.
    
    Args:
        target_total (int): total number of examples to generate (~5000)
    
    Returns:
        pd.DataFrame: balanced dataset with 10 genres
    """
    df = pd.read_csv("/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/SpotifyFeatures.csv")

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

def get_genre_mapping(df):
    genre_mapping = {genre: idx for idx, genre in enumerate(df['genre'].unique())}
    return genre_mapping

def get_graph(musics):
    features = musics.columns.values.tolist()
    cm = np.corrcoef(musics[features].values.T)
    hm = heatmap(cm, row_names=features, column_names=features, figsize=(20,10))
    plt.show()

def get_histogram(y):
    _ = plt.hist(y, bins='auto') 
    plt.title("Histogram with 'auto' bins for Music Genres")
    plt.show()  

########################################
# MLP MODEL
########################################

class MLP_Net(nn.Module):
    def __init__(self, x_means, x_deviations, n_classes=10):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations

        self.linear1 = nn.Linear(10, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.linear2(x)  # logits, no softmax here
        return x

########################################
# TRAINING LOOP
########################################

def training_loop(N_Epochs, model, loss_fn, opt, scheduler, train_dl):
    for epoch in range(N_Epochs):
        model.train()
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb.squeeze())

            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()

        if epoch % 100 == 0:
           print(epoch, "loss=", loss)

########################################
# METRICS
########################################

def print_metrics_function(y_test, y_pred):
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print("Confusion Matrix:")
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred, average='weighted', zero_division=0))

########################################
# MAIN
########################################

def main():
    np.set_printoptions(precision=4, suppress=True)
    batch_size    = 64
    learning_rate = 0.0005 ## 0.001
    N_Epochs      = 100
    epsilon = 0.0001
    
    # --- Load Data ---
    musics = get_dataset_from_kaggle_classification()
    
    genre_mapping = get_genre_mapping(musics)
    

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

    # --- Handle class imbalance ---
    class_counts = np.bincount(Y.reshape(-1))
    class_weights = 1. / np.maximum(class_counts, 1)
    weights = torch.tensor(class_weights, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    # --- Dataloaders ---
    train_dl = torch.utils.data.DataLoader(list(zip(X_train_tr, y_train_tr)), batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(list(zip(X_test_tr, y_test_tr)), batch_size=len(X_test_tr), shuffle=False)

    # --- Model ---
    n_classes = len(np.unique(Y))
    model = MLP_Net(x_means, x_deviations, n_classes=n_classes)
    opt = optim.Adam(model.parameters(), learning_rate)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.9)
    
    # --- Training ---
    training_loop(N_Epochs, model=model, loss_fn=loss_fn, opt=opt, scheduler=scheduler, train_dl=train_dl)
    torch.save(model.state_dict(), "model_classification_MLP_musics_spotify.pt")
    
    # Dummy input for ONNX (shape must match your model)
    # Here batch_size=1 and 10 features
    dummy_input = torch.randn(1, 10, dtype=torch.float32)

    # Export ONNX
    onnx_model_path = "model_classification_MLP_musics_spotify.onnx"
    model.eval()
    torch.onnx.export(
        model,                     # PyTorch model
        dummy_input,               # example input
        onnx_model_path,           # save path
        input_names=['input'],     # input name
        output_names=['output'],   # output name
        opset_version=18,# ONNX version (recent)
        external_data=False,
        dynamic_axes={
            'input': {0: 'batch_size'},   # dynamic batch_size
            'output': {0: 'batch_size'}
        },
        export_params=True     
    )

    print(f"ONNX model saved: {onnx_model_path}")
    
    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        for x_real, y_real in test_dl:
            logits = model(x_real)
            preds = torch.argmax(logits, dim=1)
            print_metrics_function(y_real.squeeze().numpy(), preds.numpy())

if __name__ == "__main__":
    main()


'''
0 loss= tensor(1.9157, grad_fn=<NllLossBackward0>)
100 loss= tensor(1.2315, grad_fn=<NllLossBackward0>)
200 loss= tensor(1.3566, grad_fn=<NllLossBackward0>)
300 loss= tensor(1.4069, grad_fn=<NllLossBackward0>)
400 loss= tensor(1.1903, grad_fn=<NllLossBackward0>)
500 loss= tensor(1.5937, grad_fn=<NllLossBackward0>)
600 loss= tensor(1.4908, grad_fn=<NllLossBackward0>)
700 loss= tensor(1.4044, grad_fn=<NllLossBackward0>)
800 loss= tensor(1.5162, grad_fn=<NllLossBackward0>)
900 loss= tensor(1.5163, grad_fn=<NllLossBackward0>)
1000 loss= tensor(1.2368, grad_fn=<NllLossBackward0>)
1100 loss= tensor(1.4748, grad_fn=<NllLossBackward0>)
1200 loss= tensor(1.1440, grad_fn=<NllLossBackward0>)
1300 loss= tensor(1.5759, grad_fn=<NllLossBackward0>)
1400 loss= tensor(1.2364, grad_fn=<NllLossBackward0>)
1500 loss= tensor(0.9045, grad_fn=<NllLossBackward0>)
1600 loss= tensor(1.5465, grad_fn=<NllLossBackward0>)
1700 loss= tensor(1.4375, grad_fn=<NllLossBackward0>)
1800 loss= tensor(1.1906, grad_fn=<NllLossBackward0>)
1900 loss= tensor(1.2191, grad_fn=<NllLossBackward0>)
2000 loss= tensor(1.0743, grad_fn=<NllLossBackward0>)
2100 loss= tensor(1.5091, grad_fn=<NllLossBackward0>)
2200 loss= tensor(1.4533, grad_fn=<NllLossBackward0>)
2300 loss= tensor(1.3588, grad_fn=<NllLossBackward0>)
2400 loss= tensor(1.4600, grad_fn=<NllLossBackward0>)
2500 loss= tensor(1.1488, grad_fn=<NllLossBackward0>)
2600 loss= tensor(1.2203, grad_fn=<NllLossBackward0>)
2700 loss= tensor(1.1693, grad_fn=<NllLossBackward0>)
2800 loss= tensor(1.2252, grad_fn=<NllLossBackward0>)
2900 loss= tensor(1.5287, grad_fn=<NllLossBackward0>)
3000 loss= tensor(1.1897, grad_fn=<NllLossBackward0>)
3100 loss= tensor(1.3496, grad_fn=<NllLossBackward0>)
3200 loss= tensor(1.5707, grad_fn=<NllLossBackward0>)
3300 loss= tensor(1.4568, grad_fn=<NllLossBackward0>)
3400 loss= tensor(1.2290, grad_fn=<NllLossBackward0>)
3500 loss= tensor(1.3691, grad_fn=<NllLossBackward0>)
3600 loss= tensor(1.2164, grad_fn=<NllLossBackward0>)
3700 loss= tensor(1.3330, grad_fn=<NllLossBackward0>)
3800 loss= tensor(1.4761, grad_fn=<NllLossBackward0>)
3900 loss= tensor(1.0878, grad_fn=<NllLossBackward0>)
Accuracy: 0.48
Confusion Matrix:
[[46 10  6  0 13  2  5  7  5  7]
 [30 12 11  1 12 10  6 13  3 10]
 [ 3  1 59  0  6 12 14  7  5  6]
 [ 2  0  0 85  1  4  2  0  0  0]
 [ 3  6  3  1 32  8  5 11  0 15]
 [ 0  3  5  1  4 62  1  8  0  0]
 [19  3 10  0  0  8 56  3  6  4]
 [ 6 10  0  0 13 26  5 26  1 12]
 [ 6  1  1  0  0  1  5  0 85  1]
 [23  7  3  0 30  6  5 14  4 16]]
Precision: 0.469
Recall: 0.479
F1-measure: 0.466
'''