from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    classifier = DecisionTreeClassifier(random_state=42) # You can replace this with your chosen algorithm
    model = classifier.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    training_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Model Accuracy: {training_accuracy}")
    print(f"Test Data Accuracy: {test_accuracy}")