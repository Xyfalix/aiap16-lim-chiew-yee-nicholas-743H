import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score

grid = {'max_depth': range(5, 10), # controls the max depth of the decision tree
         'min_samples_split': range(20, 50), # min no. of samples in partition before split
         'min_samples_leaf': range(2, 10)} # min no. of samples in leaf node.

def train_model(X_train, y_train):
    classifier = DecisionTreeClassifier(random_state=42)
    gcv = GridSearchCV(estimator=classifier, param_grid=grid)
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

    # Predictions on test data
    y_pred = model.predict(X_test)

    # Calculate precision and recall scores
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")

    # Plot the decision tree
    # plt.figure(figsize=(15, 10))
    # plot_tree(model, filled=True, feature_names=X_train.columns, class_names=["No Cancer", "Cancer"])
    # plt.show()