import torch
import onnx
import onnxruntime as ort
import numpy as np


def predict_regression(x_samples, onnx_model_path):
    """
    Prédit les valeurs continues à partir d'un modèle ONNX de régression.

    Args:
        x_samples (np.array): échantillons à prédire, shape (n_samples, n_features)
        onnx_model_path (str): chemin du modèle ONNX

    Returns:
        np.array: valeurs prédites, shape (n_samples, n_outputs)
    """
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Prédiction
    predictions = sess.run([output_name], {input_name: x_samples.astype(np.float32)})[0]

    return predictions




def main(): 
    #onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_LinearReg_movies.onnx" 
    #onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_DL_movies.onnx"
    onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_xgboost_movies_ort.onnx"
    # 4 nouveaux échantillons fictifs
    x_samples = np.array([
        [8000, 15000000, 45000000, 3.0, 2012, 105, 33.2],
    ], dtype=np.float32)
    
    

    pred = predict_regression(x_samples, onnx_model_path)
    print(f"Film : {pred}")
    
    
if __name__ == "__main__":  
    main()