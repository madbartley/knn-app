
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
import random
import pandas as pd

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
    if(len(input_chars) > 0):
        input_chars = [float(x) for x in request.form.values()]
    else:
        data = pd.read_csv('KNNGenerateSet.csv')
        data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
        data = data.dropna(axis=1)
        data = data.drop('id',axis=1)
        X = data.drop('diagnosis',axis=1)
        y = data['diagnosis']
        random_set = random.randrange(0, 13, 1)
        input_chars = X.loc[random_set, :]
        y_diagnosis = y[random_set]
        if(y_diagnosis == 0):
            y_label = "benign"
        else:
            y_label = "malignant"
    features = [np.array(input_chars)]
    scaler = load('std_scaler.bin')
    features_std = scaler.transform(features)
    predict = loaded_model.predict(features_std)
    if(predict[0] == 0):
        diagnosis = 'benign'
    else:
        diagnosis = 'malignant'

    return render_template('welcome.html', output='This tumor is likely to be {}'.format(diagnosis), zero=' {}'.format(input_chars[0]), one=' {}'.format(input_chars[1]), two=' {}'.format(input_chars[2]), three=' {}'.format(input_chars[3]), four=' {}'.format(input_chars[4]), five=' {}'.format(input_chars[5]), six=' {}'.format(input_chars[6]), seven=' {}'.format(input_chars[7]), eight=' {}'.format(input_chars[8]), nine=' {}'.format(input_chars[9]), ten=' {}'.format(input_chars[10]), eleven=' {}'.format(input_chars[11]), twelve=' {}'.format(input_chars[12]), thirteen=' {}'.format(input_chars[13]), fourteen=' {}'.format(input_chars[14]), fifteen=' {}'.format(input_chars[15]), sixteen=' {}'.format(input_chars[16]), seventeen=' {}'.format(input_chars[17]), eighteen=' {}'.format(input_chars[18]), nineteen=' {}'.format(input_chars[19]), twenty=' {}'.format(input_chars[20]), twentyone=' {}'.format(input_chars[21]), twentytwo=' {}'.format(input_chars[22]), twentythree=' {}'.format(input_chars[23]), twentyfour=' {}'.format(input_chars[24]), twentyfive=' {}'.format(input_chars[25]), twentysix=' {}'.format(input_chars[26]), twentyseven=' {}'.format(input_chars[27]), twentyeight=' {}'.format(input_chars[28]), twentynine=' {}'.format(input_chars[29]), label=' {}'.format(y_label))

if __name__ == "__main__":
    app.run()
