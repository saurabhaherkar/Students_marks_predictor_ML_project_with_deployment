import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('Students_marks_predictor_model.pkl')
df = pd.DataFrame()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    global df

    input_features = [int(i) for i in request.form.values()]
    feature_value = np.array(input_features)
    print(feature_value)
    if int(input_features[0]) < 0 or int(input_features[0]) > 24:
        return render_template('index.html', prediction_text='Please enter the value from 1 to 24 if you live in earth.')

    output = model.predict([feature_value])[0][0].round(2)

    df = pd.DataFrame([df, pd.DataFrame({'Study Hours': input_features, 'Predicted Output': [output]})])
    df.to_csv('smp_data_from_app.csv')

    return render_template('index.html', prediction_text='You will get [{}%] marks'.format(output))


if __name__ == '__main__':
    app.run(debug=True)