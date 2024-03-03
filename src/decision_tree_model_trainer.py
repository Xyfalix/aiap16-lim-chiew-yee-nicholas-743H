import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

grid = {'max_depth': range(10, 40),
         'min_samples_split': range(200, 220),
         'min_samples_leaf': [1]}

def train_model(X_train, y_train):
    classifier = DecisionTreeClassifier(random_state=42)
    gcv = GridSearchCV(estimator=classifier, param_grid=grid)
    gcv.fit(X_train, y_train)
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