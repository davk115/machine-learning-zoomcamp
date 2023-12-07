import pickle

from flask import Flask, request, jsonify
import xgboost as xgb

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('termination')

@app.route('/predict', methods=['POST'])
def predict():
    employee = request.get_json()
    X = dv.transform([employee])

    features = list(dv.get_feature_names_out())
    dmpred = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(dmpred)
    termination = y_pred >= 0.45

    result = {
        'termination_probability': float(y_pred),
        'termination': bool(termination)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)