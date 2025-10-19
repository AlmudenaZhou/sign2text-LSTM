from .train_model import train_model_workflow
from .data_preprocess import get_train_test_data


def training_pipeline_workflow(model_filename):
    X_train, X_test, y_train, y_test = get_train_test_data()
    train_model_workflow(X_train, y_train, X_test, y_test, model_filename)
