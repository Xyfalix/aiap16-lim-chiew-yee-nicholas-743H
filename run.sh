#!/bin/bash

# Uncomment the command in line 4 and run in terminal to provide execute permissions to the script
# chmod +x run.sh

# Check if the user provided a model argument
if [ -z "$1" ]; then
    echo "Usage: $0 <model>"
    echo "Available models: decision_tree, naive_bayes, random_forest"
    exit 1
fi

# Assign the provided model argument to a variable
selected_model="$1"

# Function to execute the specified model trainer script
run_model_trainer() {
    model_trainer_script="src/${1}_model_trainer.py"
    if [ -f "$model_trainer_script" ]; then
        echo "Running $1 model trainer..."
        python "$model_trainer_script"
    else
        echo "Error: $1 model trainer script not found."
    fi
}

# Run data_loader.py to load the data
python src/data_loader.py

# Run preprocessor.py to preprocess the data
python src/preprocessor.py

# Run the specified model trainer based on user input
case $selected_model in
    "decision_tree")
        run_model_trainer "decision_tree"
        ;;
    "naive_bayes")
        run_model_trainer "naive_bayes"
        ;;
    "random_forest")
        run_model_trainer "random_forest"
        ;;
    *)
        echo "Invalid model specified: $selected_model"
        echo "Available models: decision_tree, naive_bayes, random_forest"
        exit 1
        ;;
esac

# Run main.py to execute the main machine learning pipeline
python src/main.py $1