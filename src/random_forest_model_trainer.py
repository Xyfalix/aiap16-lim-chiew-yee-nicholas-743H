import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

grid = {
    'n_estimators': [10, 25, 30, 50, 100, 200],  # Number of trees in the forest
    'max_depth': [2, 3, 5, 10, 20], # controls the max depth of each decision tree
    'min_samples_leaf': [5, 10, 20, 50, 100, 200] # min no. of samples in leaf node.
}

def train_model(X_train, y_train):
    rf_classifier = RandomForestClassifier(random_state=42)
    gcv = GridSearchCV(estimator=rf_classifier, param_grid=grid, cv=4, n_jobs=-1, verbose=1, scoring="accuracy")
    gcv.fit(X_train, y_train)

    # Retrieve the best parameters
    best_params = gcv.best_params_
    print("Best Parameters:", best_params)

    model_ = gcv.best_estimator_
    print(model_)
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