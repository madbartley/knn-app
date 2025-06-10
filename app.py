
# import the flask class
from flask import Flask

# HTTP methods import
from flask import request

# For using templates
from flask import render_template

# imports for our ml script
import numpy as np
import pickle
from joblib import load

# create an instance of the Flask class, _name_ being a standin for the application's module or package
app = Flask(__name__)

# loading our model
loaded_model = pickle.load(open('knnpickle_file', 'rb'))

# the route decorator tells Flask which URL should trigger the function
# then the function returns the request in the browser - default is HTML 
@app.route("/")
def hello_world():
    return render_template('welcome.html')

@app.route("/predict", methods=['POST'])
def predict():
    input_chars = [float(x) for x in request.form.values()]
    features = [np.array(input_chars)]
    scaler = load('../std_scaler.bin')
    features_std = scaler.transform(features)
    predict = loaded_model.predict(features_std)
    if(predict[0] == 0):
        diagnosis = 'benign'
    else:
        diagnosis = 'malignant'

    return render_template('welcome.html', output='This tumor is likely to be {}'.format(diagnosis))

if __name__ == "__main__":
    app.run()
