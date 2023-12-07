import requests

url = 'http://localhost:9696/predict'

employee = {'satisfaction_rating': 0.58,
 'last_evaluation': 0.58,
 'number_of_projects': 3,
 'average_monthly_hours': 122,
 'tenure': 3,
 'work_accidents': 0,
 'promotion_last_5years': 0,
 'department': 'management',
 'salary': 'high'}


response = requests.post(url, json=employee).json()
print(response)

if response['termination'] == True:
    print('Employee is predicted to leave.')
else:
    print('Employee is not predicted to leave.')