import torch
import onnx
import onnxruntime as ort
import numpy as np

def predict_regression(x_samples: np.ndarray, onnx_model_path: str) -> np.ndarray:
    """
    Predicts continuous values using an ONNX regression model.

    Args:
        x_samples (np.ndarray): Input samples to predict, shape (n_samples, n_features)
        onnx_model_path (str): Path to the ONNX regression model

    Returns:
        np.ndarray: Predicted values, shape (n_samples, n_outputs)
    """
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Prediction
    predictions = sess.run([output_name], {input_name: x_samples.astype(np.float32)})[0]

    return predictions


def main():
    """
    Example usage of the predict_regression function with a single sample.
    """
    # Uncomment the model you want to use
    # onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_LinearReg_movies.onnx"
    # onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/model_regression_DL_movies.onnx"
    onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_regression/xgboost_movies_vote_average_ort.onnx"

    # Example sample (1 sample with 7 features)
    x_samples = np.array([
        [8000, 15000000, 45000000, 3.0, 2012, 105, 33.2],
    ], dtype=np.float32)

    pred = predict_regression(x_samples, onnx_model_path)
    print(f"Predicted movie values: {pred}")


if __name__ == "__main__":
    main()
