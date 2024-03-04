from config import Config
from data_loader import load_data
from preprocessor import preprocess_data
# from naive_bayes_model_trainer import train_model, evaluate_model
# from decision_tree_model_trainer import train_model, evaluate_model
from random_forest_model_trainer import train_model, evaluate_model
import joblib

def main():
    # Load data
    data = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)

    # # Save the model
    # joblib.dump(model, Config.MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()