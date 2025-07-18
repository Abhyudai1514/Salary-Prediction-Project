import streamlit as st
import joblib 
import numpy as np

st.title(" Salary Estimation App ")
st.divider()

years_at_company = st.number_input("Enter Years at company", min_value=0,max_value=20)
satisfaction_level = st.number_input(" Satisfaction level",min_value=0.0)
average_monthly_hours= st.number_input("Average MOnthly Hours",min_value=120,max_value=400)
X=[years_at_company,satisfaction_level,average_monthly_hours]
scaler=joblib.load("scaler.pkl")
model= joblib.load("model.pkl")
predict_button= st.button("Press for predicting the salary")
st.divider()

if predict_button:
    X1= np.array(X)
    X_array= scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    st.write(f"Salary prediction is {prediction}")

else:
    st.write("Please enter the values and press to the predict button")