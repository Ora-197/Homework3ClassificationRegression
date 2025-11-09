import torch
import onnx
import onnxruntime as ort
import numpy as np

def predict_genre(x_samples, onnx_model_path, class_labels):
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    logits = sess.run([output_name], {input_name: x_samples})[0]  # shape (1, n_classes)
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)  # softmax
    predicted_class = np.argmax(probs, axis=1)  # la classe prédite
    predicted_genre = class_labels[predicted_class[0]]
    return predicted_genre, probs



def main(): 
    onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/model_classification_MLP_musics_spotify.onnx" 
    #onnx_model_path = "/Users/yohannmeunier/Study/M1/Applied_MachineLearning/HM3/data_classification/model_classification_DL_musics_spotify.onnx"
    # 4 nouveaux échantillons fictifs
    x_samples = np.array([
        [0.12, 0.85, 210000, 0.0, 0.45, 0.3, 0.12, 0.55, 120.0, 0.65],
    ], dtype=np.float32)
    
    index_to_genre = {
    0: 'Folk',
    1: 'Indie',
    2: 'Electronic',
    3: 'Comedy',
    4: 'Children’s Music',
    5: 'Hip-Hop',
    6: 'Jazz',
    7: 'Pop',
    8: 'Soundtrack',
    9: 'Rock'
    }

    pred_class, probs = predict_genre(x_samples, onnx_model_path, index_to_genre)
    print("Classe prédite :", pred_class)
    print("Probabilités :", probs)
    
if __name__ == "__main__":  
    main()
