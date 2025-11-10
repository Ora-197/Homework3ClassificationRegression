import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlxtend.plotting import heatmap

def get_dataset_from_kaggle_regression() -> pd.DataFrame:
    """
    Loads and preprocesses the movie dataset for regression.

    Steps:
        - Load CSV
        - Filter movies with sufficient votes
        - Convert dates and extract year
        - Convert budget and revenue to numeric
        - Compute ROI
        - Keep only movies with valid numeric features
        - Select the 5000 movies with most votes
        - Remove NaNs and zero values

    Returns:
        pd.DataFrame: preprocessed movie dataset with selected numeric features
    """
    # Load CSV
    movies = pd.read_csv(
        "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/movies_metadata.csv",
        low_memory=False
    )

    # Copy dataset for safety
    movies = movies.copy()

    # Convert and filter vote_count
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce')
    movies = movies[movies['vote_count'] > 40]

    # Convert release_date and extract year
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year

    # Convert budget and revenue to numeric
    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')

    # Keep movies with valid budget and revenue
    movies = movies[(movies['budget'] > 0) & (movies['revenue'] > 0)]

    # Compute ROI
    movies['roi'] = movies['revenue'] / movies['budget']

    # Drop rows with NaN on important columns
    movies = movies.dropna(subset=['year', 'budget', 'revenue', 'roi'])

    # Select numeric features
    features_numeric = ['vote_count', 'budget', 'revenue', 'roi', 'year', 'runtime', 'popularity', 'vote_average']
    movies['vote_average'] = movies['vote_average'].round().astype(int)
    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
    movies = movies[features_numeric]

    # Keep top 5000 movies with most votes
    movies = movies.sort_values(by='vote_count', ascending=False).head(5000)

    # Remove NaNs and zero values
    movies = movies.dropna()
    movies = movies[(movies != 0).all(axis=1)]
    
    return movies


# === Linear Regression Model ===
class LinRegNet(nn.Module):
    """
    Simple linear regression neural network.
    """
    def __init__(self, x_means: torch.Tensor, x_deviations: torch.Tensor):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations
        self.linear1 = nn.Linear(7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: normalizes input and applies linear layer.
        """
        x = (x - self.x_means) / self.x_deviations
        y_pred = self.linear1(x)
        return y_pred


# === Deep Learning Model with 2 hidden layers ===
class DL_Net(nn.Module):
    """
    Deep learning regression model with 2 hidden layers.
    """
    def __init__(self, x_means: torch.Tensor, x_deviations: torch.Tensor):
        super().__init__()
        self.x_means = x_means
        self.x_deviations = x_deviations

        self.linear1 = nn.Linear(7, 10)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 6)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(6, 1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: normalizes input and applies two hidden layers with ReLU activation.
        """
        x = (x - self.x_means) / self.x_deviations
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        y_pred = self.linear3(x)
        return y_pred


# === Inference function ===
def predict_rating(
    sample_features_batch: list,
    model_path: str,
    x_means: torch.Tensor,
    x_devs: torch.Tensor
) -> np.ndarray:
    """
    Predicts movie ratings given a batch of samples and a trained PyTorch model.

    Args:
        sample_features_batch (list): list of N samples, each with 7 features
            [[vote_count, budget, revenue, roi, year, runtime, popularity], ...]
        model_path (str): path to saved .pt model
        x_means (torch.Tensor): mean values from training set
        x_devs (torch.Tensor): standard deviation values from training set

    Returns:
        np.ndarray: predicted ratings for each sample
    """
    # Load the model
    model = LinRegNet(x_means, x_devs)
    # model = DL_Net(x_means, x_devs)  # Uncomment to use deep model
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Convert to tensor
    x = torch.tensor(sample_features_batch, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(x).squeeze().numpy()

    return y_pred


# === Example usage ===
model_path = '/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_LR_movies.pt'
epsilon = 1e-4

movies = get_dataset_from_kaggle_regression()
movies_raw_data_np = movies.to_numpy()
X = movies_raw_data_np[:, :-1]
Y = movies_raw_data_np[:, 7:8]

random_seed = int(random.random() * 100)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

# Convert to correct dtype
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

# Example test samples
samples_test = [
    [15000, 30000000, 120000000, 4.0, 2016, 115, 50.3],
    [8000, 15000000, 45000000, 3.0, 2012, 105, 33.2],
    [22000, 50000000, 250000000, 5.0, 2020, 130, 72.1],
    [1200, 2000000, 5000000, 2.5, 2004, 95, 12.4]
]

predictions = predict_rating(samples_test, model_path, x_means, x_deviations)

print("\nPredicted IMDB Ratings:")
for i, p in enumerate(predictions, start=1):
    print(f"Movie {i}: {p:.2f}")


