import pickle
from flask import Flask, request, jsonify


with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

app = Flask('bank')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': round(float(y_pred), 3),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)