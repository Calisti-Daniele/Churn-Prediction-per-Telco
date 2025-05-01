# 📞 Telco Customer Churn Prediction / Previsione Abbandono Clienti Telco

> 🔍 **Chi abbandonerà il servizio e perché?**  
> 🌍 **Who will cancel and why?**

## 🎯 Objective | Obiettivo
- 🇮🇹 **Prevedere quali clienti disdiranno nei prossimi 3 mesi** e mostrare le cause principali del rischio di churn, così da attivare campagne di fidelizzazione.  
- 🇬🇧 **Predict which customers will churn within the next 3 months** and surface the main drivers of churn so the business can launch targeted retention actions.

## 📂 Dataset
- **Telco Customer Churn** (7 043 clienti, 21 variabili: contratti, servizi, pagamenti, demografia).  
  Public & open source — ideal for portfolio projects.

## 🏗️ Tech Stack
| 🔧 Tool | 💬 Description |
|---------|----------------|
| Python 3.10 | Core language |
| Pandas & Polars | Data wrangling |
| Scikit-learn / XGBoost | Baseline & boosting models |
| SHAP | Explainability (feature importance) |
| Streamlit | Lightweight web app for business users |

## 🚀 Quick Start
1. `git clone https://github.com/tuo-repo/telco-churn.git`  
2. `poetry install` (o `pip install -r requirements.txt`)  
3. `streamlit run app/app.py` — carica un CSV di clienti, ottieni la probabilità di churn e le top-features 💡

## 📈 Deliverables
- **Notebook pulito** (`notebooks/`) con EDA, feature engineering, modelli e metriche.  
- **Web app** (`app/`) che consente al reparto commerciale di caricare un file clienti e visualizzare:  
  - 🔮 *Churn probability*  
  - 🗝️ *Top-3 drivers* per cliente 

## 💡 Business Value
- 🇮🇹 **Riduci il tasso di abbandono** → +ROI campagne retention, -costi acquisizione.  
- 🇬🇧 **Lower churn rate** → higher LTV & reduced acquisition costs.

---
