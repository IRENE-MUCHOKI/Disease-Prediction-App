import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from dython.nominal import associations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier #random forest model
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "diseasepredictiondataset")

#load datasets
df_main=pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
df_description=pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
df_precaution=pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
df_severity=pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

#studying the dataset
df_main.head()
print(df_main.describe())
print(df_main.isnull().sum())
print(df_main.info())
print(df_main.value_counts())

#merge the datasets
symptom_cols = [col for col in df_main.columns if col.startswith("Symptom_")]

#train test split
X = pd.get_dummies(df_main[symptom_cols])
y = df_main["Disease"]
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

#evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#save the model
with open(os.path.join(DATA_DIR, "disease_prediction_model.pkl"), "wb") as file:
    pickle.dump(model, file)

#add the other csv files
def get_description(disease):
    row = df_description[df_description["Disease"] == disease]

    if not row.empty:
        return row.iloc[0]["Description"]
    else:
        return "Description not found."

def get_precautions(disease):
    row = df_precaution[df_precaution["Disease"] == disease]

    if not row.empty:
        return [
            row.iloc[0]["Precaution_1"],
            row.iloc[0]["Precaution_2"],
            row.iloc[0]["Precaution_3"],
            row.iloc[0]["Precaution_4"]
        ]
    else:
        return []


def predict_disease(symptoms):
    patient = pd.DataFrame(0, index=[0], columns=X.columns)

    for symptom in symptoms:
        symptom = symptom.strip().lower()

        for col in X.columns:
            if col.lower() == symptom:
                patient[col] = 1
                break

    disease = model.predict(patient)[0]

    description = get_description(disease)
    precautions = get_precautions(disease)

    return disease, description, precautions
