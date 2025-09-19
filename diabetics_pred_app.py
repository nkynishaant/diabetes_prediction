# streamlit ui
import numpy as np
import pickle
import streamlit as st
import os  # for handling relative paths

# Load the saved model using a relative path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.sav")
loaded_model = pickle.load(open(MODEL_PATH, "rb"))

# Function for prediction
def diabetes_prediction(input_data):
    # Convert input list to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    
    # Reshape the array for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Return result
    if prediction[0] == 1:
        return 'The person is diabetic'
    else:
        return 'The person is non diabetic'


def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # User input
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the person')

    # Button for prediction
    if st.button('Diabetes Test Result'):
        # Convert inputs to float before passing to the model
        diagnosis = diabetes_prediction([
            float(Pregnancies),
            float(Glucose),
            float(BloodPressure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ])
        st.success(diagnosis)


if __name__ == '__main__':
    main()
