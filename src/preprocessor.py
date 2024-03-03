# preprocessor.py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

def preprocess_data(data):
    # drop duplicate Patient ID entries
    data.drop_duplicates(subset=['ID'], inplace=True)

    # Create a boolean mask for negative Age values
    negative_age_mask = data['Age'] < 0

    # Update the 'Age' column by using numpy's abs function to convert all -ve ages to +ve
    data.loc[negative_age_mask, 'Age'] = np.abs(data.loc[negative_age_mask, 'Age'])

    # Replace "MALE" with "Male" and "FEMALE" with "Female" in Gender column
    data["Gender"] = data["Gender"].replace({"MALE": "Male", "FEMALE": "Female"})

    # Remove row where 'Gender' value is 'NAN', since there is only 1 entry
    data = data[data['Gender'] != 'NAN']

    # Assumption made that null values should most likely by No, since COPD is a serious condition and it would most likely be documented if it occurred.
    data.loc[:, 'COPD History'] = data['COPD History'].fillna('No')

    # Remove the 3 rows with NaN values in Air Pollution Exposure from dataset
    data = data.dropna(subset=['Air Pollution Exposure'])

    # Create new column Weight Difference
    data['Weight Difference'] = data['Current Weight'] - data['Last Weight']

    # Convert Not Applicable values for both columns to 0, Convert Still Smoking values in Stop Smoking Column to 2023.
    data['Start Smoking'] = data['Start Smoking'].replace('Not Applicable', 0)

    data['Stop Smoking'] = data['Stop Smoking'].replace('Not Applicable', 0)
    data['Stop Smoking'] = data['Stop Smoking'].replace('Still Smoking', 2024)

    # Convert Year string values into numerical values
    data['Start Smoking'] = pd.to_numeric(data['Start Smoking'], errors='coerce')
    data['Stop Smoking'] = pd.to_numeric(data['Stop Smoking'], errors='coerce')

    # Create new column "Smoking Duration (Years)"
    data["Smoking Duration (Years)"] = data["Stop Smoking"] - data["Start Smoking"]

    # There are 1026 entires of NaN for this column, which I will fill with No. The assumption here is that there would be documented record if person has taken Bronchodilators
    data.loc[:, 'Taken Bronchodilators'] = data['Taken Bronchodilators'].fillna('No')

    # Select independent variables to use (X) and dependent variable (y)
    columns_to_drop = ["Lung Cancer Occurrence", "ID", "Age", "Last Weight", "Current Weight", "Start Smoking", "Stop Smoking", "Frequency of Tiredness", "Dominant Hand"]
    X = data.drop(columns_to_drop, axis=1, inplace=False)
    y = data["Lung Cancer Occurrence"]

    print("This is the cleaned up data")
    
    label_encoder = LabelEncoder()

    # Apply LabelEncoder to Air Pollution Exposure which is ordinal
    X["Air Pollution Exposure"] = label_encoder.fit_transform(X["Air Pollution Exposure"])

    # Apply one-hot encoding to other non-ordinal categoral columns
    X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

    # convert true false values to boolean
    X = X.astype(int)

    all_columns = X.columns.tolist()

    # Display the list of columns
    # print(all_columns)
    # print(X.head())
    # print(X.describe())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test