import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlxtend.plotting import heatmap
import random
import matplotlib.pyplot as plt


def get_dataset_from_kaggle_regression() -> pd.DataFrame:
    """
    Load and preprocess the movie dataset for linear regression.

    Steps:
        - Load CSV
        - Filter movies with sufficient votes
        - Convert release_date to datetime and extract year
        - Convert budget and revenue to numeric
        - Keep only movies with valid budget and revenue
        - Compute ROI (revenue / budget)
        - Drop rows with NaN in important columns
        - Keep selected numeric features
        - Keep top 5000 movies with most votes
        - Remove NaNs and zero values

    Returns:
        pd.DataFrame: preprocessed dataset with numeric features
    """
    movies = pd.read_csv(
        "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/movies_metadata.csv",
        low_memory=False
    )

    movies = movies.copy()
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce')
    movies = movies[movies['vote_count'] > 40]

    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year

    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')
    movies = movies[(movies['budget'] > 0) & (movies['revenue'] > 0)]

    movies['roi'] = movies['revenue'] / movies['budget']
    movies = movies.dropna(subset=['year', 'budget', 'revenue', 'roi'])

    features_numeric = ['vote_count', 'budget', 'revenue', 'roi', 'year', 'runtime', 'popularity', 'vote_average']
    movies['vote_average'] = movies['vote_average'].round().astype(int)
    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
    movies = movies[features_numeric]

    movies = movies.sort_values(by='vote_count', ascending=False).head(5000)
    movies = movies.dropna()
    movies = movies[(movies != 0).all(axis=1)]

    return movies


def get_graph(movies: pd.DataFrame):
    """
    Plot a heatmap of correlations between numeric features.

    Args:
        movies (pd.DataFrame): dataset containing numeric features
    """
    features = movies.columns.values.tolist()
    cm = np.corrcoef(movies[features].values.T)
    heatmap(cm, row_names=features, column_names=features, figsize=(20, 10))
    plt.show()


# === Linear Regression Model ===
class LinRegNet(nn.Module):
    """
    Simple linear regression model using PyTorch.
    """
    def __init__(self, x_means: torch.Tensor, x_deviations: torch.Tensor):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations
        self.linear1 = nn.Linear(7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: normalize input and apply linear layer.
        """
        x = (x - self.x_means) / self.x_deviations
        y_pred = self.linear1(x)
        return y_pred


def training_loop(N_Epochs: int, model: nn.Module, loss_fn, opt: torch.optim.Optimizer, train_dl: DataLoader):
    """
    Training loop for PyTorch models.

    Args:
        N_Epochs (int): number of epochs
        model (nn.Module): PyTorch model
        loss_fn: loss function (e.g., F.mse_loss)
        opt (torch.optim.Optimizer): optimizer
        train_dl (DataLoader): training data loader
    """
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 20 == 0:
            print(epoch, "loss=", loss)


def main():
    # Parameters
    batch_size = 16
    learning_rate = 0.005
    N_Epochs = 100
    epsilon = 1e-4
    np.set_printoptions(precision=4, suppress=True)

    # Load dataset
    movies = get_dataset_from_kaggle_regression()
    movies_raw_data_np = movies.to_numpy()
    X = movies_raw_data_np[:, :-1]
    Y = movies_raw_data_np[:, 7:8]

    random_seed = int(random.random() * 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

    # Convert data type
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    X_train_tr = torch.from_numpy(X_train)
    X_test_tr = torch.from_numpy(X_test)
    y_train_tr = torch.from_numpy(y_train)
    y_test_tr = torch.from_numpy(y_test)

    x_means = X_train_tr.mean(0, keepdim=True)
    x_deviations = X_train_tr.std(0, keepdim=True) + epsilon

    train_ds = TensorDataset(X_train_tr, y_train_tr)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    # Initialize model
    model = LinRegNet(x_means, x_deviations)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = F.mse_loss

    # Training
    training_loop(N_Epochs, model, loss_fn, opt, train_dl)
    torch.save(model.state_dict(), "model_regression_LR_movies.pt")

    # Export to ONNX
    dummy_input = torch.randn(1, 7, dtype=torch.float32)
    onnx_model_path = "model_regression_LinearReg_movies.onnx"
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=18,
        external_data=False,
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        export_params=True
    )
    print(f"ONNX model saved: {onnx_model_path}")

    # Evaluate model
    y_pred_test = model(X_test_tr)
    print("Testing R²:", r2_score(y_test_tr.numpy(), y_pred_test.detach().numpy()))

    # Print first 15 predictions vs real values
    list_preds = []
    list_reals = []
    for i in range(min(15, len(X_test_tr))):
        np_real = y_test_tr[i].detach().numpy()
        np_pred = y_pred_test[i].detach().numpy()
        print("************************************")
        print("pred, real")
        print((np_pred, np_real))
        list_preds.append(np_pred[0])
        list_reals.append(np_real[0])


if __name__ == "__main__":
    main()
 
    
    
    
'''
0 loss= tensor(29.0793, grad_fn=<MseLossBackward0>)
20 loss= tensor(0.6251, grad_fn=<MseLossBackward0>)
40 loss= tensor(0.4811, grad_fn=<MseLossBackward0>)
60 loss= tensor(0.5703, grad_fn=<MseLossBackward0>)
80 loss= tensor(0.4574, grad_fn=<MseLossBackward0>)

Testing R**2:  0.3166987895965576
************************************
pred, real
(array([5.757], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.4643], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.5905], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.3463], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.5014], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.049], dtype=float32), array([5.], dtype=float32))
************************************
pred, real
(array([6.5554], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.2831], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([5.8434], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.3008], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([5.9513], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([7.2378], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.5953], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.8231], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([7.8572], dtype=float32), array([8.], dtype=float32))
'''