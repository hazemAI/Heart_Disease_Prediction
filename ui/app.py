import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

MODEL_PATH = '../models/final_model.pkl'

FEATURE_ORDER = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'fbs', 'exang',
    'cp', 'restecg', 'slope', 'ca', 'thal'
]

CATEGORICAL_FEATURES = ['cp', 'restecg', 'slope', 'ca', 'thal']


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def main():
    st.set_page_config(page_title='Heart Disease Predictor', page_icon='❤️', layout='centered')
    st.title('Heart Disease Risk Prediction')
    st.write('Use the form below to input clinical data and estimate the probability of heart disease.')

    model = load_model()
    if model is None:
        st.warning('Model not found. Please run the training notebooks to generate ../models/final_model.pkl')

    with st.form('input_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age (years)', min_value=18, max_value=100, value=54)
            sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: f"{'Female' if x==0 else 'Male'} ({x})")
            cp = st.selectbox('Chest Pain Type (cp)', options=[1, 2, 3, 4])
            trestbps = st.number_input('Resting Blood Pressure (trestbps, mm Hg)', min_value=80, max_value=220, value=130)
            chol = st.number_input('Serum Cholesterol (chol, mg/dl)', min_value=100, max_value=700, value=240)
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', options=[0, 1])
        with col2:
            restecg = st.selectbox('Resting ECG (restecg)', options=[0, 1, 2])
            thalach = st.number_input('Max Heart Rate (thalach)', min_value=60, max_value=240, value=150)
            exang = st.selectbox('Exercise Induced Angina (exang)', options=[0, 1])
            oldpeak = st.number_input('ST Depression (oldpeak)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox('Slope of Peak ST (slope)', options=[1, 2, 3])
            ca = st.selectbox('Major Vessels Colored (ca)', options=[0, 1, 2, 3])
            thal = st.selectbox('Thalassemia (thal)', options=[3, 6, 7])

        submitted = st.form_submit_button('Predict')

    if submitted:
        # Construct input DataFrame
        data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        X = pd.DataFrame([data], columns=FEATURE_ORDER)

        if model is None:
            st.stop()

        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0][1]
            else:
                # Fallback using decision_function scaled to [0,1]
                scores = model.decision_function(X)
                proba = (scores[0] - scores.min()) / (scores.max() - scores.min() + 1e-9)
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.subheader('Prediction')
        st.metric(label='Heart Disease Probability', value=f"{proba*100:.1f}%")
        st.write(f"Predicted class: {'Disease Present (1)' if pred==1 else 'No Disease (0)'}")

        with st.expander('Input Summary'):
            st.json(data)

    st.markdown('---')
    st.caption('Model expects the 13 clinical features from the UCI Heart Disease (Cleveland) subset. Categorical features are safely handled with one-hot encoding and unknown categories ignored.')


if __name__ == '__main__':
    main()
