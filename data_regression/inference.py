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

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
###########
import numpy as np
import torch
import pandas as pd
import sklearn
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from mlxtend.plotting import heatmap
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

## coefficient of determination 
from sklearn.metrics import r2_score

import xgboost as xgb
import onnxruntime as rt
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import regressor
def get_dataset_from_kaggle_regression():
    # 0️⃣ Charger le fichier CSV
    movies = pd.read_csv("/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/movies_metadata.csv", low_memory=False)

    # 1️⃣ Copier le dataset pour sécurité
    movies = movies.copy()

    # 2️⃣ Convertir et filtrer vote_count
    movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce')
    movies = movies[movies['vote_count'] > 40]

    # 3️⃣ Convertir la date et extraire l'année
    movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
    movies['year'] = movies['release_date'].dt.year

    # 4️⃣ Convertir budget et revenue
    movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce')
    movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce')

    # 5️⃣ Garder films avec budget et revenue valides
    movies = movies[(movies['budget'] > 0) & (movies['revenue'] > 0)]

    # 6️⃣ Calculer ROI
    movies['roi'] = movies['revenue'] / movies['budget']

    # 7️⃣ Supprimer les lignes avec NaN sur les colonnes importantes
    movies = movies.dropna(subset=['year', 'budget', 'revenue', 'roi'])

    # 8️⃣ Sélection des features numériques
    features_numeric = ['vote_count', 'budget', 'revenue', 'roi', 'year', 'runtime', 'popularity', 'vote_average']
    movies['vote_average'] = movies['vote_average'].round().astype(int)
    movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce')
    movies = movies[features_numeric ]  # ajouter title pour TF-IDF

    # 9️⃣ Prendre les 5000 films avec le plus de votes
    movies = movies.sort_values(by='vote_count', ascending=False).head(5000)

    # 🔟 Supprimer NaN et lignes avec 0
    movies = movies.dropna()
    movies = movies[(movies != 0).all(axis=1)]
    
    return movies
# === Définition du MLP identique à celui entraîné ===
## Linear Regression

class LinRegNet(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(7, 1)
        
    ## perform inference
    def forward(self, x):
        
        x = (x - self.x_means) / self.x_deviations
        
        y_pred = self.linear1(x)
        ## return torch.round( y_pred )
        return y_pred
## Deep Learning with 2 hidden layers

class DL_Net(nn.Module):
    ## init the class
    def __init__(self, x_means, x_deviations):
        super().__init__()
        
        self.x_means      = x_means
        self.x_deviations = x_deviations
        
        self.linear1 = nn.Linear(7, 10)
        self.act1    = nn.ReLU()
        self.linear2 = nn.Linear(10, 6)
        self.act2    = nn.ReLU()
        self.linear3 = nn.Linear(6, 1)
        self.dropout = nn.Dropout(0.25)
        
    ## perform inference
    def forward(self, x):
        
        x = (x - self.x_means) / self.x_deviations
        
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        ## x = self.dropout(x)
        y_pred = self.linear3(x)
        
        ## return torch.round( y_pred )
        return y_pred

# === Fonction d'inférence ===
import torch
import numpy as np

def predict_rating(sample_features_batch, model_path, x_means, x_devs):
    """
    sample_features_batch : liste de N samples, chacun avec 7 features :
        [
            [vote_count, budget, revenue, roi, year, runtime, popularity],
            ...
        ]

    model_path : chemin du .pt sauvegardé
    x_means, x_devs : tensors venant du training (à passer ou recharger)
    """

    # Charger le modèle
    model = LinRegNet(x_means, x_devs)
    #model = DL_Net(x_means, x_devs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Convertir vers tenseur batch
    x = torch.tensor(sample_features_batch, dtype=torch.float32)

    # Normalisation (sera appliquée dans forward également si tu le gardes)
    # Ici on ne la refait pas si ton modèle la fait déjà dans forward()
    # Sinon décommente :
    # x = (x - x_means) / x_devs

    with torch.no_grad():
        y_pred = model(x).squeeze().numpy()

    return y_pred





model_path = '/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_LR_movies.pt'
#model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_DL_movies.pt"
epsilon = 0.0001
movies = get_dataset_from_kaggle_regression()
movies_raw_data_np = movies.to_numpy()
X = movies_raw_data_np[:, :-1]
#Between 3 and 9
Y = movies_raw_data_np[:, 7:8]
random_seed = int( random.random() * 100 )     ## 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
## fix data type

X_train = X_train.astype(  np.float32  )
X_test  = X_test.astype(   np.float32  )
y_train = y_train.astype(  np.float32  )
y_test  = y_test.astype(   np.float32  )
X_train_tr = torch.from_numpy(X_train)
X_test_tr  = torch.from_numpy(X_test)
y_train_tr = torch.from_numpy(y_train)
y_test_tr  = torch.from_numpy(y_test)
x_means      = X_train_tr.mean(0, keepdim=True ) 
x_deviations = X_train_tr.std( 0, keepdim=True) + epsilon

samples_test = [
    [15000, 30000000, 120000000, 4.0, 2016, 115, 50.3],
    [8000, 15000000, 45000000, 3.0, 2012, 105, 33.2],
    [22000, 50000000, 250000000, 5.0, 2020, 130, 72.1],
    [1200, 2000000, 5000000, 2.5, 2004, 95, 12.4]
]






predictions = predict_rating(samples_test, model_path, x_means, x_deviations)

print("\n✅ Prédictions IMDB :")
for i, p in enumerate(predictions, start=1):
    print(f"Film {i}: {p:.2f}")

