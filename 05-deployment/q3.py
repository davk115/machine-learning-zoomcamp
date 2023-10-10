import pickle
import requests


URL="https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2023/05-deployment/homework"
model_url = URL+"/model1.bin"
dv_url = URL+"/dv.bin"


model = requests.get(model_url, allow_redirects=True)
open('model1.bin', 'wb').write(model.content)

dv = requests.get(dv_url, allow_redirects=True)
open('dv.bin', 'wb').write(dv.content)


with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

client = {"job": "retired", "duration": 445, "poutcome": "success"}

X = dv.transform([client])
y_pred = round(model.predict_proba(X)[0, 1], 3)

print(y_pred)