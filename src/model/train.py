# Import libraries
import argparse
import glob
import os

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from azureml.core import Run, Model

# define functions

def main(args):
    # TO DO: enable autologging
    mlflow.sklearn.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    model = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    register_model(model, args.training_data)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data

def split_data(df):
    # Assume the last column is the target variable
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]   # Target

    # Split into train and test sets (80/20 split)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        # train model
        model = LogisticRegression(C=1/reg_rate, solver="liblinear")
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        print(f"Model trained with accuracy: {accuracy:.4f}")

def register_model(model, training_data_path):
    """Registers the trained model in Azure ML"""
    model_path = "outputs/trained_model.pkl"
    os.makedirs("outputs", exist_ok=True)
    mlflow.sklearn.save_model(model, model_path)

    run.upload_file(name="trained_model.pkl", path_or_stream=model_path)

    # Register the model in Azure ML
    registered_model = run.register_model(
        model_name="my_model",
        model_path="trained_model.pkl",
        tags={"training_data": training_data_path},
        description="Logistic Regression Model for CI/CD pipeline"
    )

    print(f"✅ Model registered: {registered_model.name}, Version: {registered_model.version}")


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str, required=True)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
