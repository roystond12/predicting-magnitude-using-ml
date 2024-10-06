from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('earthquake_magnitude_xgboost2_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x.replace(',', '')) for x in request.form.values()]
    while len(features) < 3551:
        features.append(0.0) 

    final_features = [np.array(features)]

    # Make prediction
    prediction = model.predict(final_features)
    predicted_value = prediction[0]

    return render_template('index.html', 
                           prediction_text=f'Predicted Earthquake Magnitude: {predicted_value:.2f}',
                           prediction=predicted_value)


if __name__ == '__main__':
    app.run(debug=True)
