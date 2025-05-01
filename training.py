# 📦 Librerie
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # per salvare scaler

# 📥 Caricamento dati già codificati
df = pd.read_csv("datasets/EDA_data.csv", index_col=0)

# 🎯 Target
target = df['Churn']
df = df.drop('Churn', axis=1)

# 🛠️ Feature Engineering su df codificato
df['TotalServices'] = df[
    [
        'PhoneService_Yes', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'OnlineSecurity_Yes', 'OnlineBackup_Yes',
        'DeviceProtection_Yes', 'TechSupport_Yes',
        'StreamingTV_Yes', 'StreamingMovies_Yes'
    ]
].sum(axis=1)

df['AvgMonthlyCharge'] = (df['TotalCharges'] / df['tenure']).replace([np.inf, -np.inf], 0).fillna(0)

def tenure_group(t):
    if t < 12: return 0
    elif t < 24: return 1
    elif t < 48: return 2
    elif t < 60: return 3
    else: return 4
df['TenureGroup'] = df['tenure'].apply(tenure_group)

df['IsLongContract'] = df[['Contract_One year', 'Contract_Two year']].sum(axis=1) > 0
df['HasTechSupport'] = df['TechSupport_Yes']
df['HasStreaming'] = df[['StreamingTV_Yes', 'StreamingMovies_Yes']].sum(axis=1) > 0

# 🧪 Separazione train/test
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, stratify=target, random_state=42)

# 📏 Normalizzazione solo delle colonne numeriche
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'TotalServices']

scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 💾 Salva lo scaler per futuri preprocessing
joblib.dump(scaler, "models/scaler.pkl")

# 🚀 Addestramento CatBoost
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    class_weights=[1, 3],
    random_state=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

# 📈 Valutazione
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("AUC-ROC:", roc_auc_score(y_test, y_prob))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("CatBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 💾 Salvataggio del modello
model.save_model("models/catboost_model.cbm")
joblib.dump(X_train.columns.tolist(), "models/columns.pkl")
