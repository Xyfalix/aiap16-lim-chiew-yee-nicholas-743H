from data_loader import load_data
from preprocessor import preprocess_data
import sys

def main(model_name):
    # import the specified model trainer module
    try:
        print(f"Importing {model_name}_model_trainer")
        model_trainer_module = __import__(f"{model_name}_model_trainer", globals(), locals(), ["train_model", "evaluate_model"], 0)
    except ImportError:
        print(f"Error: {model_name}_model_trainer module not found.")
        sys.exit(1)

    # Load data
    data = load_data()

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train model
    model = model_trainer_module.train_model(X_train, y_train)

    # Evaluate model
    model_trainer_module.evaluate_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    # Check if a model name is provided as a command-line argument
    # print(sys.argv)
    # print(len(sys.argv))
    if len(sys.argv) != 2:
        print("Usage: python main.py <model>")
        sys.exit(1)

    # Assign the provided model name to a variable
    model_name = sys.argv[1]

    # Call the main function with the specified model name
    main(model_name)