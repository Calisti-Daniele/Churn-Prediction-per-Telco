# Librerie essenziali
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
df = pd.read_csv("datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Prima occhiata
print("Shape:", df.shape)
print("Column types:\n", df.dtypes)
print(df.head())

# Distribuzione Churn
sns.countplot(x='Churn', data=df)
plt.title("Distribuzione variabile target 'Churn'")
plt.show()

# Missing e dati inconsistenti
print(df.isnull().sum())
print("\nValori vuoti (spazi):")
print((df == " ").sum())

# Pulizia campo TotalCharges (contiene stringhe vuote)
df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)

# Rimuovi righe con TotalCharges mancanti
df = df[df['TotalCharges'].notna()]

# Reset dell'indice
df = df.reset_index(drop=True)

# Converte SeniorCitizen in variabile categorica
df['SeniorCitizen'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

# Variabili categoriche da convertire in dummy (one-hot)
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('customerID')  # Non utile per il modello
df = df.drop('customerID', axis=1)


# Salvo la target per dopo
target = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.drop('Churn', axis=1)

# 🔁 One-hot encoding sulle feature categoriche
cat_cols = df.select_dtypes(include='object').columns.tolist()
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Riaggiungo il target alla fine
df_encoded['Churn'] = target

# Dataset pronto per feature engineering / modellazione
print("✅ Dataset pulito e codificato. Shape finale:", df_encoded.shape)
print(df_encoded.head())


df_encoded.to_csv("datasets/EDA_data.csv")