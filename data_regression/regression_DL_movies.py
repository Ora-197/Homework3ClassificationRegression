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

def get_graph(movies):
    features = movies.columns.values.tolist()
    cm = np.corrcoef( movies[features].values.T   )
    hm = heatmap(cm, row_names=features, column_names=features, figsize=(20,10))
    plt.show()

###################################

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




def training_loop( N_Epochs, model, loss_fn, opt, train_dl ):
    
    for epoch in range(N_Epochs):
        for xb, yb in train_dl:
            
            y_pred = model(xb)
            loss   = loss_fn(y_pred, yb)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if epoch % 20 == 0:
            print(epoch, "loss=", loss)

def main(): 
    ################################
    #Parameters
    batch_size    = 16
    learning_rate = 0.005 ## 0.001
    N_Epochs      = 100

    epsilon = 0.0001
    np.set_printoptions(precision=4, suppress=True)
    #################################
    
    
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
    train_ds = TensorDataset( X_train_tr, y_train_tr  )
    train_dl = DataLoader( train_ds, batch_size, shuffle=True  )
    
    ################################
    #def core_function Linear Regression
    ################################
    
   
    model = DL_Net( x_means, x_deviations  )
    opt     = torch.optim.Adam(    model.parameters(), lr=learning_rate )
    loss_fn = F.mse_loss
    training_loop(  N_Epochs, model, loss_fn, opt, train_dl)
    torch.save(model.state_dict(), "model_regression_DL_movies.pt")
    
    # Export ONNX
    # Dummy input pour ONNX (la forme doit correspondre à ton modèle)
    # Ici batch_size=1 et 7 features
    dummy_input = torch.randn(1, 7, dtype=torch.float32)
    onnx_model_path = "model_regression_DL_movies.onnx"
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
    
    ###############################
    #Evaluate the model
    ###############################
    
    y_pred_test = model( X_test_tr )
    print( "Testing R**2: ", r2_score(  y_test_tr.numpy(),  y_pred_test.detach().numpy()     )  ) 
    list_preds = []
    list_reals = []
    for i in range(15):
    #for i in range(len(X_test_tr)):
        print("************************************")
        print("pred, real")
        np_real =   y_test_tr[i].detach().numpy()
        np_pred = y_pred_test[i].detach().numpy()
        print(( np_pred  , np_real))
        list_preds.append(np_pred[0])
        list_reals.append(np_real[0])

if __name__ == "__main__":
    main()  

'''
0 loss= tensor(0.6944, grad_fn=<MseLossBackward0>)
20 loss= tensor(0.3652, grad_fn=<MseLossBackward0>)
40 loss= tensor(0.7056, grad_fn=<MseLossBackward0>)
60 loss= tensor(0.7884, grad_fn=<MseLossBackward0>)
80 loss= tensor(0.4971, grad_fn=<MseLossBackward0>)
Testing R**2:  0.4242740869522095
************************************
pred, real
(array([7.1904], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([7.843], dtype=float32), array([8.], dtype=float32))
************************************
pred, real
(array([7.0433], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([7.3757], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.7245], dtype=float32), array([8.], dtype=float32))
************************************
pred, real
(array([6.1241], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.0947], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([7.5377], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([7.885], dtype=float32), array([8.], dtype=float32))
************************************
pred, real
(array([6.5246], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([5.6881], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.3245], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.2339], dtype=float32), array([6.], dtype=float32))
************************************
pred, real
(array([6.178], dtype=float32), array([7.], dtype=float32))
************************************
pred, real
(array([6.9467], dtype=float32), array([7.], dtype=float32))
'''