import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

def train_model(X_train, y_train):
    classifier = GaussianNB()
    params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
    gcv = GridSearchCV(estimator=classifier, param_grid=params_NB)
    gcv.fit(X_train, y_train)

     # Retrieve the best parameters
    best_params = gcv.best_params_
    print("Best Parameters:", best_params)

    model_ = gcv.best_estimator_
    optimized_model = model_.fit(X_train, y_train)
    return optimized_model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Model Accuracy: {training_accuracy}")
    print(f"Test Data Accuracy: {test_accuracy}")

    # Plot the decision tree
    # plt.figure(figsize=(15, 10))
    # plot_tree(model, filled=True, feature_names=X_train.columns, class_names=["No Cancer", "Cancer"])
    # plt.show()