# Heart Disease ML Pipeline

A complete ML pipeline on the UCI Heart Disease dataset: preprocessing, PCA, feature selection, supervised and unsupervised learning, model tuning, and a Streamlit app for inference.

- Dataset: https://archive.ics.uci.edu/ml/datasets/heart+disease
- App: `ui/app.py` (predicts heart-disease risk from 13 clinical features)

## Demo

https://github.com/user-attachments/assets/762154a9-070b-4335-8f40-a43b2167be64

## Project Structure
```
.
├─ data/
│  └─ heart_disease.csv
├─ notebooks/
│  └─ 01_data_preprocessing.ipynb
│  └─ 02_pca_analysis.ipynb
│  └─ 03_feature_selection.ipynb
│  └─ 04_supervised_learning.ipynb
│  └─ 05_unsupervised_learning.ipynb
│  └─ 06_hyperparameter_tuning.ipynb
├─ models/
│  └─ final_model.pkl         # Trained sklearn pipeline
├─ results/
│  └─ evaluation_metrics.txt  # Metrics summaries
├─ ui/
│  └─ app.py                  # Streamlit UI
├─ deployment/
│  └─ ngrok_setup.txt         # ngrok quick start
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Environment Setup
```bash
pip install -r requirements.txt
```
Python packages used: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib, streamlit, xgboost.

## Reproduce the Pipeline (notebooks)
Open and run the notebooks in order (Jupyter/VS Code):
- `notebooks/01_data_preprocessing.ipynb`
- `notebooks/02_pca_analysis.ipynb`
- `notebooks/03_feature_selection.ipynb`
- `notebooks/04_supervised_learning.ipynb`
- `notebooks/05_unsupervised_learning.ipynb`
- `notebooks/06_hyperparameter_tuning.ipynb`

Alternatively (headless execution example for the first notebook):
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_preprocessing.ipynb
# repeat for the remaining notebooks in order
```
The trained pipeline is saved to `models/final_model.pkl` and consumed by the UI.

## Run the Streamlit App (Windows)
```powershell
python -m streamlit run ui/app.py --server.port 8502 --server.address 0.0.0.0
```
Open:
- Local: http://127.0.0.1:8502
- LAN:   http://<your LAN IP>:8502

### Input Features expected by the UI
Order used by the model pipeline:
```
age, trestbps, chol, thalach, oldpeak, sex, fbs, exang, cp, restecg, slope, ca, thal
```
Categorical features internally one-hot encoded: `cp, restecg, slope, ca, thal`.

## Expose Publicly with ngrok (optional)
First make sure you installed ngrok and have a valid token from https://ngrok.com/. Then run the following commands:
```powershell
# once
ngrok config add-authtoken YOUR_NGROK_TOKEN
# start tunnel
ngrok http http://127.0.0.1:8502
```
Grab the https URL from the ngrok console, or programmatically:
```powershell
(Invoke-RestMethod http://127.0.0.1:4040/api/tunnels).tunnels `
  | Where-Object {$_.proto -eq 'https'} `
  | Select-Object -ExpandProperty public_url
```

## Results (summary)
See `results/evaluation_metrics.txt` for full details. Highlights:
- Baseline best (SVM): Accuracy 0.902, ROC AUC 0.960
- Tuned best (Random Forest Grid): Accuracy 0.902, ROC AUC 0.951
- PCA retained 95% variance with 18 components; feature selection highlighted `thal`, `cp`, `ca`, `thalach`, `oldpeak` as strong predictors.

## Troubleshooting
- Model not found in the app: ensure `models/final_model.pkl` exists (re-run training scripts).
- ngrok ERR_NGROK_8012 (502): ensure Streamlit runs and tunnel targets IPv4 explicitly:
  - `python -m streamlit run ui/app.py --server.port 8502 --server.address 0.0.0.0`
  - `ngrok http http://127.0.0.1:8502`
- ngrok ERR_NGROK_3200 (offline): you opened a URL from a previous session. Use the current URL shown in ngrok console.
- Port busy: switch to another port (e.g., 8503) and update both commands accordingly.

## Acknowledgments
- UCI Machine Learning Repository — Heart Disease dataset.
