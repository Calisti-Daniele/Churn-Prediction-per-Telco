import streamlit as st
import pandas as pd
import joblib
import os

from catboost import CatBoostClassifier
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -----------------------------
# CONFIG
st.set_page_config(page_title="📉 Churn Predictor", layout="centered")

# -----------------------------
# LOAD MODELS
scaler = joblib.load("models/scaler.pkl")
model = CatBoostClassifier()
model.load_model("models/catboost_model.cbm")
columns = joblib.load("models/columns.pkl")

# -----------------------------
# LLM via HuggingFace Inference API
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
client = InferenceClient(api_key=HF_API_KEY, provider="novita")

def get_churn_explanation(probability, features):
    prompt = f"""
    Sei un consulente esperto in customer retention per aziende di telecomunicazioni. Il cliente ha una probabilità del {probability}% di abbandonare il servizio.
    Caratteristiche del cliente:
    {features}

    Rispondi in italiano. Fornisci:
    1. Una breve analisi della probabilità di churn.
    2. I fattori principali che influenzano la decisione.
    3. Tre suggerimenti pratici per ridurre il rischio di abbandono.
    Aggiungi emoji dove appropriato per migliorare la leggibilità.
    """

    messages = [{"role": "user", "content": prompt}]

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=messages,
            max_tokens=800,
            temperature=0.6,
            top_p=0.9
        )
        return response.choices[0].message["content"]
    except Exception as e:
        st.warning("⚠️ LLaMA non è disponibile, sto usando Mistral come backup.")
        try:
            response = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.3",
                messages=messages,
                max_tokens=800,
                temperature=0.6,
                top_p=0.9
            )
            return response.choices[0].message["content"]
        except Exception as e:
            return "❌ Errore nel generare la spiegazione. Riprova più tardi."

# -----------------------------
# UI
st.title("📞 Telco Customer Churn Predictor")
st.markdown("""
Questa app predice la probabilità che un cliente **abbandoni il servizio** (churn), in base ai suoi dati. 
Carica un file CSV con le informazioni dei clienti e visualizza i risultati.
""")

uploaded_file = st.file_uploader("📤 Carica un file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Encoding
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = df.drop(columns=[col for col in ['customerID', 'Churn'] if col in df.columns], errors='ignore')
    cat_cols = [col for col in cat_cols if col not in ['customerID', 'Churn']]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    # Feature Engineering
    df['TotalServices'] = df[[
        'PhoneService_Yes', 'MultipleLines_Yes', 'InternetService_Fiber optic',
        'OnlineSecurity_Yes', 'OnlineBackup_Yes', 'DeviceProtection_Yes',
        'TechSupport_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes']].sum(axis=1)
    df['AvgMonthlyCharge'] = (df['TotalCharges'] / df['tenure']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['TenureGroup'] = df['tenure'].apply(lambda t: 0 if t < 12 else 1 if t < 24 else 2 if t < 48 else 3 if t < 60 else 4)
    df['IsLongContract'] = df.get('Contract_One year', 0) + df.get('Contract_Two year', 0) > 0
    df['HasTechSupport'] = df.get('TechSupport_Yes', 0)
    df['HasStreaming'] = df.get('StreamingTV_Yes', 0) + df.get('StreamingMovies_Yes', 0) > 0

    # Normalizzazione
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'TotalServices']
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Predizioni
    probs = model.predict_proba(df)[:, 1]
    predictions = (probs >= 0.5).astype(int)
    results = df.copy()
    results['Churn Prediction'] = predictions
    results['Probabilità di Churn'] = (probs * 100).round(2)

    st.subheader("📋 Risultati delle Predizioni")
    st.dataframe(results[['Probabilità di Churn', 'Churn Prediction']])

    # Spiegazione
    st.subheader("🧠 Spiegazione personalizzata")
    customer_idx = st.number_input("Scegli il numero del cliente (riga)", min_value=0, max_value=len(df)-1, step=1)
    if st.button("🔎 Genera spiegazione con LLaMA"):
        features_str = df.iloc[customer_idx].to_dict()
        prob = results['Probabilità di Churn'].iloc[customer_idx]
        with st.spinner("Sto generando la spiegazione..."):
            explanation = get_churn_explanation(prob, features_str)
        st.markdown(
            f"""
            <div style='padding:1rem;border-radius:10px;background:#f0f0f0;border:1px solid #ccc;'>
            <h4>🤖 Spiegazione del LLM</h4>
            <p>{explanation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("⬆️ Carica un file per iniziare.")
    with open("datasets/for_prediction.csv", "rb") as f:
        st.download_button("📥 Scarica un file di esempio", f, file_name="sample_churn_input.csv", mime="text/csv")
