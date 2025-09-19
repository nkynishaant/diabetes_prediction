#streamlit ui
import numpy as np
import pickle#loading the save model
import streamlit as st
#loading the saved model
loaded_model=pickle.load(open('/Users/nishantkumar/Downloads/AI/trained_model.sav','rb'))

#creating a function for prediction


def diabetes_prediction(input_data):
    
    input_data_as_numpy_array=np.asarray(input_data)#asarray is use to tranform      list into numpyarray
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)#this is the          diameter we giving to tell the model that we are predicting only one instance
    #standardise the data 
    
   
    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==1):#[0]this represents the first value in prediction
       return('The person is diabetic')
    else:
       return('The person is non diabetic')


def main():
    #giving title
    st.title('Diabetes Prediction Web App')
    #getting the input data from the user
   
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood pressure value')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI Value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function Value')
    Age=st.text_input('Age of the person')



    #code for prediction
    diagnosis=''#empty string whose values are in the func diabetes_prediction
    #creating a button for prediction

    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis)
if __name__=='__main__':
    main()