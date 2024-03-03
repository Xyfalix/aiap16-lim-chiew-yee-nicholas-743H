from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config import Config

def train_model(X_train, y_train):
    model = RandomForestClassifier()  # You can replace this with your chosen algorithm
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")