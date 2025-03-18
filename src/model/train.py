import os
import argparse
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn

def log_confusion_matrix_image(cm, labels, normalize=False, log_name='confusion_matrix', title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Save confusion matrix image
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join('outputs', f'{log_name}.png'))
    mlflow.log_artifact(os.path.join('outputs', f'{log_name}.png'))

def main(args):
    mlflow.sklearn.autolog()
    os.makedirs('outputs', exist_ok=True)

    X, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=223)
    
    with mlflow.start_run():
        svm_model = SVC(kernel=args.kernel, C=args.penalty, gamma='scale')
        svm_model.fit(x_train, y_train)
        
        svm_predictions = svm_model.predict(x_test)
        
        cm = confusion_matrix(y_test, svm_predictions)
        labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        log_confusion_matrix_image(cm, labels, normalize=False, log_name='confusion_matrix_unnormalized')
        log_confusion_matrix_image(cm, labels, normalize=True, log_name='confusion_matrix_normalized')
        
        joblib.dump(svm_model, os.path.join('outputs', 'model.pkl'))
        mlflow.log_artifact(os.path.join('outputs', 'model.pkl'))
        mlflow.sklearn.log_model(svm_model, "model")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type for SVM')
    parser.add_argument('--penalty', type=float, default=1.0, help='Penalty parameter for SVM')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
