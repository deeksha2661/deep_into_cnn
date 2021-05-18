import numpy as np
import csv
from sklearn.metrics import f1_score
 
from train import *

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv". It should be created in the same directory where this code file is present.
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X


def predict_target_values(test_X):
	X = np.genfromtxt('train_X_knn.csv', delimiter = ',', dtype = np.float64, skip_header = 1)
	Y = np.genfromtxt('train_Y_knn.csv', delimiter = ',', dtype = np.float64)
	return np.array(classify_points(X, Y, test_X, 7))
    

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    pred_Y = pred_Y.reshape(len(pred_Y), 1)
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(pred_Y)
        csv_file.close()

def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    pred_Y = pred_Y.astype(int)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")

def validate():
    pred_Y = np.genfromtxt("predicted_test_Y_knn.csv", delimiter=',', dtype=np.int)
    actual_Y = np.genfromtxt("test_Y_knn.csv", delimiter=',', dtype=np.int)
    #print(pred_Y)
    #print(actual_Y)
    weighted_f1_score = f1_score(actual_Y, pred_Y, average = 'weighted')
    print("Weighted F1 score", weighted_f1_score)
    

if __name__ == "__main__":
    test_X_file_path = "test_X_knn.csv"
    predict(test_X_file_path)
    validate()
