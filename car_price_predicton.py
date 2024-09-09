#!/usr/bin/env python  
# coding: utf-8  

from flask import Flask, render_template, request  
import numpy as np  
import joblib  
import pandas as pd  

app = Flask(__name__)  

# Load the model and scaler  
model_path = 'C:/Users/User/Desktop/furkan/AutoScout/final_model.pkl'  
model = joblib.load(model_path)  

# Function to make predictions  
def make_prediction(sample_obs):  
    # Convert sample_obs to a DataFrame and apply one-hot encoding  
    sample_df = pd.DataFrame([sample_obs], columns=["make_model", "hp_kW", "km", "age", "Gearing_Type"])  
    sample_df = pd.get_dummies(sample_df)  
    
    # Align the DataFrame with the model's training features  
    sample_df = sample_df.reindex(columns=model.feature_names_in_, fill_value=0)  
    
    # Make the prediction  
    prediction = model.predict(sample_df)  
    return prediction  

@app.route("/", methods=["GET", "POST"])  
def home():  
    prediction = None  
    result_message = ""  
    if request.method == "POST":  
        try:  
            features = [  
                str(request.form["make_model"]),  
                float(request.form["hp_kW"]),  
                float(request.form["km"]),  
                float(request.form["age"]),  
                str(request.form["Gearing_Type"]),  
            ]  
           
            # Make a prediction  
            prediction = make_prediction(features)  
            result_message = f"Estimated price of your car with the given values is: {int(prediction[0])}"  
        except Exception as e:  
            result_message = f"Error: {e}"  # Display the error message  

    return render_template("form.html", prediction=result_message)  

if __name__ == "__main__":  
    app.run(debug=True)