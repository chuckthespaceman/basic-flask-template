#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peter Simeth's basic flask pretty youtube downloader (v1.3)
https://github.com/petersimeth/basic-flask-template
© MIT licensed, 2018-2023
"""

from flask import Flask, render_template, request
import pickle
import logging
from part_module import PART, Node
import pandas as pd

DEVELOPMENT_ENV = True

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app_data = {
    "name": "Peter's Starter Template for a Flask Web App",
    "description": "A basic Flask app using bootstrap for layout",
    "author": "Peter Simeth",
    "html_title": "Peter's Starter Template for a Flask Web App",
    "project_name": "Starter Template",
    "keywords": "flask, webapp, template, basic",
}

result = {
    "prediction": "" 
}

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'PART':
            return PART
        if name == 'Node':
            return Node
        return super().find_class(module, name)

with open('./templates/PART_model.pkl', 'rb') as f:
    part_data = CustomUnpickler(f).load()

@app.route("/")
def index():
    return render_template("index.html", app_data=app_data, result="")

@app.route("/about")
def about():
    return render_template("about.html", app_data=app_data)

@app.route("/submit", methods=['POST'])
def submit():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    
    # Do something with the data (e.g., print it)
    print("N:", N)
    print("P:", P)
    print("K:", K)
    print("Temperature:", temperature)
    print("Humidity:", humidity)
    print("pH:", ph)
    print("Rainfall:", rainfall)
    
    # Optionally, you can perform further processing or calculations here
        
    data = {
        'N': [float(N)],
        'P': [float(P)],
        'K': [float(K)],
        'temperature': [float(temperature)],
        'humidity': [float(humidity)],
        'ph': [float(ph)],
        'rainfall': [float(rainfall)],
    }

    df = pd.DataFrame(data)

    # Use the loaded model to make prediction
    prediction = part_data.predict(df)
    print("prediction:", prediction)

    result['prediction'] = prediction
    # Redirect or render a response as needed
    return render_template('index.html', app_data=app_data, result=result)
     

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
