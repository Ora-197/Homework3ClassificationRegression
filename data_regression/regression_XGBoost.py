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



def main(): 
    ################################
    #Parameters
    batch_size    = 16
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
    
    regressor = xgb.XGBRegressor(

        n_estimators=100,
        reg_lambda=1,
        gamma=0,
        max_depth=3
    )
    
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    print(y_pred)
    
    print("Testing R**2 : ", r2_score(y_test, y_pred))
    
    initial_types = [(
          'float_input',
          FloatTensorType(  [None, 7 ]  )

    )]
    
    
    

    # Définir les types d'entrée (nombre de features = X_train.shape[1])
    initial_types = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

    # Convertir le modèle XGBoost en ONNX
    onnx_model = onnxmltools.convert_xgboost(regressor, initial_types=initial_types)

    # Sauvegarder le modèle ONNX
    onnxmltools.utils.save_model(onnx_model, 'xgboost_movies_vote_average_ort.onnx')

    # Charger le modèle ONNX pour l'inférence
    sess = rt.InferenceSession('xgboost_movies_vote_average_ort.onnx')

    # Récupérer les noms d'entrée et de sortie
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    # Faire l'inférence sur X_test (attention à float32)
    pred_ort = sess.run([label_name], {input_name: X_test.astype(np.float32)})

    # Afficher les résultats
    print("Predictions ONNX (XGBoost):", pred_ort[0][0:15])  # Afficher les 10 premières prédictions

    

if __name__ == "__main__":
    main()  
    

'''
[5.8711 6.9537 6.5682 6.7793 5.7946 6.9851 6.7044 6.2996 5.9135 6.3567
 6.6413 5.0094 6.6682 7.1441 6.2083 6.266  5.9908 6.9058 5.6897 6.0725
 5.732  7.0286 6.8883 6.1824 6.4065 6.7264 5.1359 6.2792 6.688  6.2444
 6.9182 6.3225 6.4683 7.0371 6.7857 5.8128 6.4224 6.1027 6.5181 6.2014
 6.1904 6.2539 5.4181 6.3783 5.9882 6.9896 5.9973 6.2632 5.5243 5.9859
 6.4446 7.4034 8.2578 6.6245 6.2552 6.2859 5.7278 6.9612 6.1506 6.3988
 6.2795 5.7572 6.3903 6.0018 6.0806 7.6955 6.2367 6.2053 6.9229 6.72
 5.7787 5.9939 5.9017 5.8322 5.702  6.209  7.0314 5.0743 7.4101 7.6229
 5.9619 6.203  6.5575 6.5753 6.3266 5.7658 7.3479 6.0582 6.261  7.7043
 6.2982 5.4561 5.6525 8.0224 6.7493 6.1214 6.4649 7.8188 5.7382 5.9918
 5.6419 6.2207 6.7902 6.0045 5.6788 6.0406 5.5448 6.4818 7.0827 5.4682
 7.0048 6.0777 6.5326 7.5811 6.5813 6.4401 5.3809 6.6522 6.5125 7.0457
 6.2568 6.3077 5.9667 7.1487 5.5144 6.1989 7.4505 5.7001 6.5712 6.4668
 5.8732 6.1798 5.7014 6.0846 6.6159 5.9846 7.2925 6.717  7.4463 7.1493
 6.2135 5.2016 5.732  6.2555 7.1225 5.9657 5.6134 5.9739 6.0085 6.0717
 6.2871 8.1117 5.8692 7.2012 5.497  5.8171 6.266  6.1454 6.262  5.7419
 6.5887 5.5464 7.2922 6.732  6.8176 7.273  6.6854 6.8064 6.9157 7.4965
 6.9747 6.8305 6.7156 5.4007 6.561  6.4177 6.7187 6.6077 7.4464 7.3035
 7.3137 6.2452 7.8041 5.8195 6.9845 5.0538 6.7516 6.485  5.3358 5.876
 6.4757 6.0999 6.3285 6.0298 6.5675 6.654  7.426  5.3075 6.4486 5.847
 5.4566 6.9227 6.5725 6.2209 5.8851 6.5822 6.097  6.8688 6.2947 6.1867
 6.0209 7.0203 6.1477 6.473  5.8676 6.2224 6.0138 6.7686 6.3234 6.5249
 6.3423 5.9557 5.3864 6.2096 7.2183 7.1213 6.1929 6.3111 6.7326 6.109
 5.8829 5.498  7.4001 6.7288 6.6002 7.2262 5.7677 6.5887 6.5922 6.9865
 6.3903 6.5077 7.1767 6.5038 7.2306 6.1219 6.2023 6.2587 6.3183 7.2486
 5.7212 5.3387 7.2747 6.9977 6.6211 5.9645 6.687  5.9728 7.6295 7.0491
 6.1443 5.6759 6.6242 6.8639 7.1829 6.9197 6.2951 6.1654 5.7155 8.2933
 7.6015 6.634  6.4818 6.3822 5.3884 6.5803 6.4618 6.1413 6.843  6.2712
 5.4761 5.8688 6.0027 5.8345 7.1082 6.0285 7.0117 6.0814 6.118  5.5833
 6.5875 7.0076 6.3564 6.5071 6.8684 5.71   6.0222 7.8415 8.0694 5.7314
 6.07   6.4544 6.4363 5.4608 6.3676 6.3898 6.2615 5.6928 7.2053 7.7671
 5.7112 6.03   7.1088 6.2    7.234  6.2542 6.252  5.9915 6.2965 6.3564
 6.1909 5.8218 6.6852 4.7753 7.4183 5.2695 5.3328 7.3176 6.5371 6.9547
 6.2133 6.3467 7.1878 5.903  6.8881 6.6118 7.1049 6.3107 6.8724 7.0673
 5.826  6.3069 6.5311 6.9056 6.7906 5.6948 6.0844 5.6972 6.8056 6.0147
 6.0646 6.4143 6.1974 5.9048 6.1044 7.3467 6.6502 5.0533 6.393  7.2952
 7.197  6.7161 5.1729 6.6614 6.6552 6.4364 6.0018 6.4257 5.5964 6.3573
 6.9162 6.5778 6.2638 5.7434 6.7254 6.2751 7.5167 7.1622 6.2247 6.1148
 7.8396 6.3646 5.7262 6.823  6.2623 6.0148 6.1906 5.5619 6.075  6.3966
 6.5331 5.8593 8.3939 5.841  6.6313 6.3982 5.852  5.2222 6.4083 5.432
 5.752  7.6989 7.043  7.3057 6.0768 5.9303 5.5492 6.0365 5.9937 5.9919
 5.793  6.3074 6.2457 5.7919 6.3021 5.8402 6.3957 6.1266 6.4744 6.0623
 7.9081 6.6608 6.0336 6.3306 5.923  6.4276 5.4659 6.4838 6.2335 6.3203
 5.4936 5.5524 7.1003 5.5231 6.4482 7.0656 5.7587 6.4473 5.9055 5.559
 6.2088 7.3061 6.3872 6.8515 5.4799 6.4149 7.9402 7.766  7.5202 5.8675
 6.0575 6.6417 6.4563 6.2427 6.6226 6.2141 8.2617 6.1681 6.2395 5.4317
 5.521  6.1351 6.2005 5.7657 5.5668 6.1674 6.9122 6.6036 6.0981 6.0937
 6.6498 5.8826 6.1521 5.9616 6.1674 6.3717 6.7338 6.2698 7.2395 6.5585
 6.1629 6.0308 6.2404 6.8443 6.5367 7.5155 6.7957 7.4509 6.0606 6.4474
 6.6121 7.3721 5.1468 6.3449 7.2897 5.6626 6.1304 6.2434 7.1136 6.3289
 6.5741 7.3152 7.3612 6.0022 5.4013 6.2388 5.4952 6.5105 6.3028 6.5233
 6.0022 6.8576 5.6084 5.2066 7.0763 5.6149 6.0026 6.3197 6.4127 5.9315
 7.7504 6.2965 5.7511 5.9561 5.8727 6.8669 5.0743 6.88   6.1542 6.6419
 7.3352 6.1387 5.4845 6.4607 5.353  6.1375 6.9765 6.4906 6.0135 6.4427
 6.1532 6.3591 7.4113 7.0703 6.7559 6.4306 7.1367 5.7456 7.4773 7.7419
 6.3437 6.4084 6.9583 6.0712 6.3549 5.4759 6.7888 5.8506 6.1724 6.0337
 6.4444 6.3353 6.2663 6.1572 6.1022 6.6339 6.4133 8.2565 6.7298 6.2136
 6.1508 6.7732 8.3019 5.8469 6.6034 5.5452 5.6351 6.5603 6.723  6.0443
 5.2894 7.2202 5.9699 5.5735 4.7979 6.2051 5.8554 5.5734 6.1392 5.88
 6.6991 6.0727 5.4847 5.7185 6.0507 6.246  7.1543 5.7272 5.5275 5.9819
 6.5931 6.8207 7.3904 5.9784 5.6372 5.5216 6.7133 6.8166 6.7061 6.5709
 5.5684 6.5962 6.0398 6.1691 5.8561 5.6987 6.6458 6.4012 6.4571 5.6457
 6.1708 7.2445 7.0261 6.3918 5.7725 6.257  5.7332 6.4694 5.7771 5.873
 6.2381 6.5545 6.4393 5.2247 6.5959 6.7968 6.0007 7.3897 6.9025 5.8698
 5.7567 6.2216 6.0438 6.35   6.1694 6.3716 6.6635 6.1301 5.9124 5.8266
 6.0701 7.4878 5.6679 6.9308 6.4697 6.1854 7.3413 6.3093 6.8585 6.0555
 7.5013 5.7437 6.78   6.9781 6.2553 6.9238 6.4006 6.5936 5.8994 6.0457
 7.119  6.6644 5.9141 6.9815 6.3572 6.3291 6.6375 7.2382 5.9185 6.9919
 6.0219 7.1226 5.9868 6.4113 6.2169 7.4003 5.907  6.3881 7.295  6.3392
 6.4379 5.4365 5.7541 5.4789 6.8878 6.3668 5.5121 5.9691 6.6436 7.4622
 6.8222 5.9619 7.0539 6.7767 5.8139 6.8064 8.0804 6.2494 6.5655 6.9454
 5.7914 7.4979 6.9625 6.1135 6.5238 6.8967 5.5805 6.5299 5.8412 5.9108
 6.4195 5.9552 6.1469 5.4895 5.9128 6.8471 6.5088 7.4254 6.1255 6.0136
 6.1303 6.3245 5.9025 5.753  6.3506 6.4232 6.4407 5.4424 6.1267 6.7382
 5.4391 6.772  6.352  6.3505 6.3407 6.319  6.0246 5.8377 5.7789 7.4558
 7.2086 5.3185 6.9664 5.9983 4.8819 6.2657 5.5428 6.1175 5.5394 6.073
 6.8118 6.3505 6.3469 6.0243 7.8755 7.9049 5.7137 6.5573 6.0944 5.3389
 6.5231 6.4229 6.682  6.1331 6.0182 6.3202 6.3546 6.3773 6.0439 6.1381
 5.6847 7.2827 6.4046 5.0793 6.4521 6.1497 6.4493 6.2595 6.5009 5.8893
 6.5591 6.8431 6.0601 6.6839 6.1643 7.0069 6.1905 5.2892 5.9358 6.6593
 5.9843 6.2935 6.8859 5.1732 6.5762 6.394  6.9642 6.7966 6.4327 7.2213
 7.3249 6.0507 6.23   6.6557 6.9459 7.224  7.1992 6.4453 6.375  5.756
 6.7508 7.3255 6.6856 6.4854 6.0222 5.3463 6.4312 5.8175 6.7433 6.6517
 7.3032 6.77   5.6962 5.8464 6.0331 6.9588 6.3419 5.3784 6.1276 6.1072
 7.7223 5.7236 6.0888 6.8626 5.9275 6.4063 6.2224 5.4616 6.0894 6.3977
 7.2739 6.4661 5.9727 6.0181 6.7486 7.2034 6.7463 6.6433 6.3937 6.0799
 7.229  6.0307 7.1254 5.9809 5.4107 6.4987 6.0772 6.4032 6.7285 6.2709
 6.1224 6.0551 6.3808 6.5585 6.597  6.7395 6.4497 6.2144 5.4962 6.0779
 6.3074 6.5043 6.6579 6.3895 6.3724 6.5639 8.3685 5.7277 6.2245 5.7286
 6.3944 5.5093 6.6091 5.9709 6.2458 6.4254 6.6088 7.4777 6.348  6.3056]
Testing R**2 :  0.4409562945365906
Predictions ONNX (XGBoost): [[5.8711]
 [6.9537]
 [6.5682]
 [6.7793]
 [5.7946]
 [6.9851]
 [6.7044]
 [6.2996]
 [5.9135]
 [6.3567]
 [6.6413]
 [5.0094]
 [6.6682]
 [7.1441]
 [6.2083]]

'''