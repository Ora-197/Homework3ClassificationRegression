import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlxtend.plotting import heatmap
import random
import matplotlib.pyplot as plt

import xgboost as xgb
import onnxruntime as rt
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType


def get_dataset_from_kaggle_regression() -> pd.DataFrame:
    """
    Load and preprocess the movie dataset for regression.

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


def main():
    """
    Train an XGBoost regressor on movie data, evaluate it, 
    convert it to ONNX, and perform inference with ONNX runtime.
    """
    batch_size = 16
    epsilon = 1e-4
    np.set_printoptions(precision=4, suppress=True)

    # Load dataset
    movies = get_dataset_from_kaggle_regression()
    movies_np = movies.to_numpy()
    X = movies_np[:, :-1]
    Y = movies_np[:, 7:8]

    random_seed = int(random.random() * 100)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_seed)

    # Convert data type
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Train XGBoost regressor
    regressor_model = xgb.XGBRegressor(
        n_estimators=100,
        reg_lambda=1,
        gamma=0,
        max_depth=3
    )
    regressor_model.fit(X_train, y_train)

    # Predictions with XGBoost
    y_pred = regressor_model.predict(X_test)
    print("Predictions (XGBoost):", y_pred[:15])
    print("Testing R²:", r2_score(y_test, y_pred))

    # Convert XGBoost model to ONNX
    initial_types = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = onnxmltools.convert_xgboost(regressor_model, initial_types=initial_types)
    onnxmltools.utils.save_model(onnx_model, 'xgboost_movies_vote_average_ort.onnx')

    # ONNX inference
    sess = rt.InferenceSession('xgboost_movies_vote_average_ort.onnx')
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_ort = sess.run([label_name], {input_name: X_test.astype(np.float32)})
    print("Predictions ONNX (XGBoost):", pred_ort[0][:15])


if __name__ == "__main__":
    main()

    

'''
/M1/Applied_MachineLearning/HM3/data_regression/regression_XGBoost.py
[5.8014 5.6376 6.885  6.0486 6.2524 6.3523 5.8538 6.3711 5.5353 6.2798
 7.0303 5.5476 7.057  6.797  6.0047 5.8343 5.5837 6.6381 5.9827 5.0421
 6.4578 5.8749 5.8527 6.7363 6.2405 6.5691 5.704  6.7303 7.9359 6.3962
 6.062  6.7793 5.4409 5.5762 5.406  5.9356 7.7373 6.5487 6.3181 5.4055
 5.6326 6.8652 6.0858 6.3573 7.0531 5.8819 5.753  6.3186 6.3664 6.5322
 6.1467 5.7196 7.0305 5.6273 6.5313 5.6496 6.0957 6.4679 6.6243 5.9487
 7.897  6.7755 5.6754 6.1561 5.3499 6.4792 6.5995 5.9753 6.2558 6.893
 6.3225 6.709  5.6725 5.4939 5.9658 6.3532 6.3768 6.0173 6.0636 6.9203
 5.2181 6.3054 6.1253 5.8692 6.195  5.7204 5.8438 5.4013 5.7167 8.0339
 5.5168 6.3023 5.6016 7.0737 7.1544 6.9085 5.6242 6.1906 7.0033 7.3718
 6.2685 7.5095 5.515  6.7646 6.4026 5.2791 6.192  5.7332 6.6932 7.3494
 7.3054 5.7563 6.2376 5.618  5.9187 5.4277 6.5175 6.0463 5.3369 7.0816
 6.7152 6.4915 6.8747 6.2528 6.5294 5.8825 7.7759 6.6355 6.4486 6.246
 6.3038 6.0271 7.4699 6.0195 6.4811 7.1998 5.1428 6.5134 6.7375 6.1352
 5.6759 5.8397 6.641  7.3407 6.8448 5.8834 6.1972 6.7757 6.6949 5.8355
 6.2755 5.4064 7.3371 7.3841 5.6142 6.8142 6.1715 6.1871 6.1567 7.191
 5.5577 5.2379 5.745  6.9819 6.6274 6.5474 5.4351 6.5926 5.3856 5.8261
 6.0563 6.8801 5.9374 6.1837 6.6763 5.9603 6.9902 7.4525 6.5022 5.5998
 6.8951 6.3149 7.0894 6.237  7.3808 6.5343 6.0181 6.442  6.9248 6.0341
 7.4728 6.1395 6.5007 7.9196 5.9803 6.9324 6.5063 6.8245 7.2078 6.2091
 6.4183 6.5432 7.3398 6.5501 6.5582 6.8557 6.5766 6.3751 6.5099 6.621
 5.5747 7.5015 5.9011 6.1007 5.6799 6.3478 7.2233 7.0097 6.8195 5.8592
 6.6589 5.7046 7.5222 6.6849 5.8939 7.7865 6.3459 5.8926 7.3583 6.0045
 6.6734 6.162  6.7328 6.159  5.6228 6.7081 6.081  5.1607 6.3959 6.0947
 6.9957 5.6944 6.0849 6.7685 6.2781 6.2894 6.1062 6.4021 7.1496 6.0004
 5.7959 5.9429 6.6603 6.7045 6.8756 5.7248 5.7296 5.8153 6.5423 7.5553
 8.3984 7.7093 5.8555 7.3288 7.8418 6.7654 5.4949 6.9381 5.8723 6.9799
 6.5937 6.5082 5.7843 6.8186 6.1997 5.4439 6.4932 5.5955 6.6497 5.1781
 5.7081 6.3595 6.2577 5.8133 6.5007 6.0469 6.8768 5.5927 7.718  5.9614
 5.834  6.656  5.4598 6.4046 5.7608 5.8863 6.5879 6.0468 7.7606 6.6377
 7.4272 7.3528 5.8863 7.2986 5.758  5.7144 6.0273 6.467  6.1974 7.4299
 6.0805 7.5222 6.1926 6.2296 6.1452 5.9008 6.1125 6.014  6.9444 6.7228
 5.629  5.9488 6.0018 6.6577 5.9738 5.8301 6.5375 6.4149 5.6258 6.265
 5.8182 6.8037 6.4196 6.3177 7.4827 5.5325 6.4032 6.0978 6.2502 7.2297
 5.5004 5.5248 5.2798 5.9413 6.9974 7.465  5.9627 5.9775 6.5981 6.4446
 5.9787 6.8063 5.7942 6.5825 6.7416 6.5198 6.3995 6.6144 6.9449 6.4584
 6.5295 6.021  6.9133 7.526  7.6218 6.9547 7.1677 6.3889 7.8315 6.1277
 5.9374 6.0836 6.5267 5.9341 6.2298 6.4566 6.5206 6.1815 6.1173 5.9427
 6.9642 6.7059 6.1251 6.2778 5.627  5.5102 6.7458 6.2538 5.9353 6.4444
 8.1459 6.7674 6.159  6.7258 5.7977 5.6534 6.0673 7.0661 5.723  6.5327
 7.0174 6.7091 7.4415 6.2877 5.8113 5.9992 7.4076 5.4688 6.7147 5.9997
 5.7502 6.419  6.8037 5.7895 5.5323 6.695  5.8538 6.0858 5.6415 6.0474
 4.7198 6.3516 6.2462 6.1865 5.9894 7.4374 5.8814 8.3219 5.9187 6.5544
 5.544  5.9251 6.6503 6.215  5.823  5.3666 6.3525 6.3008 6.0246 7.2897
 5.865  6.6027 7.3634 6.5856 5.6643 5.9218 6.5803 7.6635 5.6144 6.6739
 7.2194 6.2574 6.089  6.0726 7.3894 6.8487 5.7229 5.9613 5.8271 7.4988
 7.2445 5.269  6.1751 5.8136 6.0768 6.0209 7.2852 7.4369 6.6105 6.2389
 7.269  6.6605 6.994  6.3821 6.1674 5.9066 6.4155 5.7293 6.2094 6.1888
 5.2758 6.1225 6.0275 6.3817 5.9283 5.3619 7.0092 5.6074 5.6803 7.8023
 7.2545 6.4009 5.8589 6.4824 6.4168 5.7698 6.1792 6.2137 5.3754 5.4624
 7.7718 6.6511 6.4798 5.8973 6.4619 5.7395 6.7011 7.5489 6.4485 6.4511
 7.2081 6.6109 7.8891 6.4364 7.5441 6.478  6.0063 6.2687 5.9852 7.6695
 5.7629 6.9769 6.6438 6.6215 6.1457 6.1339 6.7646 7.0348 7.3264 7.4241
 6.2609 5.695  6.434  5.7408 6.4444 6.7292 5.7713 7.0547 5.6143 6.826
 6.1268 6.7327 5.5947 6.6159 6.0546 6.1967 6.9439 6.1241 6.224  5.9903
 5.9912 7.8047 6.2334 7.0552 6.1563 7.7277 6.6469 6.4142 6.4171 6.5556
 5.5396 6.0566 7.9945 6.5894 6.3024 6.0994 6.0297 5.9057 7.8994 6.1802
 6.4305 5.281  5.7894 6.0271 5.6955 7.2856 6.0783 5.9849 6.6263 6.2401
 6.5597 6.6646 6.6529 5.4884 6.9022 6.853  5.3879 6.4027 6.5029 6.5747
 7.2739 5.2814 5.6411 5.8868 6.5836 7.1256 6.2145 5.5242 5.9532 7.7607
 6.2036 6.4075 5.5392 7.1436 6.477  6.9903 6.15   7.1356 7.3217 5.6976
 6.1128 6.5067 5.2866 5.9676 5.8597 6.0268 6.6258 5.9527 5.8242 7.124
 5.7111 6.2576 6.9707 5.7255 6.0456 6.5966 6.1911 6.5516 5.4918 7.153
 6.271  6.1061 6.6145 5.7405 5.4793 6.2095 5.9691 6.044  6.2964 6.9738
 7.5371 5.9066 7.8501 6.205  6.8455 5.9821 5.8996 6.0195 5.525  6.4056
 6.1712 5.8896 5.9721 7.0597 8.0382 6.119  6.2146 6.0821 5.5251 6.3044
 7.5909 7.5981 6.4032 6.3931 5.6903 7.5759 6.7986 6.2598 5.5487 6.1581
 6.4468 6.5209 5.813  6.3562 6.7685 6.9613 7.4586 6.4256 6.082  6.0362
 6.6254 6.8749 7.3019 5.8104 5.8053 6.6306 7.1167 5.7681 6.672  7.2431
 5.4844 6.0261 6.7615 6.4691 5.7761 6.0508 5.527  6.1296 6.5724 6.1708
 6.4428 6.4925 6.3211 6.3681 6.357  6.3775 5.6937 5.8413 6.094  5.9929
 5.3992 8.0576 5.7915 6.2018 5.8879 6.6735 5.5244 5.9724 6.5394 7.2623
 5.6796 5.8389 6.0129 5.376  5.7137 5.8397 5.5502 5.7444 5.7894 7.438
 5.979  6.7392 6.0367 6.4966 7.2834 5.9379 6.3325 6.7572 5.6699 6.0564
 7.1379 7.3987 7.594  6.1384 5.2396 5.9525 6.3273 7.2203 6.9957 5.1759
 6.0072 7.232  6.3706 7.5152 6.4501 6.484  6.6402 5.3291 6.662  6.6591
 7.8769 6.0905 6.9385 6.4862 5.5669 8.2219 6.99   5.8238 5.8693 6.9046
 5.8864 5.6337 7.2352 6.0264 7.3948 5.8979 6.5136 6.0078 6.1557 5.8953
 6.4383 5.8015 5.8037 7.3634 7.067  5.393  5.4108 5.47   6.5187 6.1458
 6.1543 6.2514 6.1887 6.0221 7.1109 7.329  6.9368 6.2009 5.4824 4.6602
 6.7193 7.6372 6.6245 5.682  7.1716 5.5877 5.7524 6.4936 7.5878 6.4673
 6.2544 5.6627 6.7842 6.4295 6.8481 7.6841 6.2704 6.8969 6.6298 6.1704
 6.5107 6.0617 5.3774 6.3861 6.3115 7.3111 6.2282 6.1069 6.3614 5.3316
 7.1422 6.0098 7.4375 6.2751 5.9374 6.4409 6.0092 5.5189 6.4333 6.6823
 7.0792 5.9774 6.3311 6.2657 6.1266 6.4238 6.7791 5.9767 7.405  5.4131
 6.1574 5.6362 6.7098 6.5913 5.4318 6.3043 7.2093 6.3599 6.5472 7.173
 7.073  6.3114 6.0737 6.0573 5.4278 5.6156 6.6219 6.2848 6.1415 6.2077
 6.1732 6.5103 6.5067 5.5261 6.5247 6.43   6.2206 5.8191 5.9054 6.4377
 6.5153 6.475  7.2043 6.4854 6.1072 7.2763 6.4717 6.6275 7.1091 7.8355
 6.2833 6.048  6.4769 6.0464 6.0422 5.6865 6.2175 5.9841 5.4817 7.1494]
Testing R**2 :  0.4356662631034851
Predictions ONNX (XGBoost): [[5.8014]
 [5.6376]
 [6.885 ]
 [6.0486]
 [6.2524]
 [6.3523]
 [5.8538]
 [6.3711]
 [5.5353]
 [6.2798]
 [7.0303]
 [5.5476]
 [7.057 ]
 [6.797 ]
 [6.0047]]
'''