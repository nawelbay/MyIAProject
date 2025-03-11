import json
import requests

data = [
    [4.3, 3.0, 1.1, 0.1],
    [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.4],
    [5.4, 3.9, 1.3, 0.4],
    [5.1, 3.5, 1.4, 0.3],
    [5.7, 3.8, 1.7, 0.3],
    [5.1, 3.8, 1.5, 0.3],
    [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.4],
    [4.6, 3.6, 1.0, 0.2],
    [5.1, 3.3, 1.7, 0.5],
    [4.8, 3.4, 1.9, 0.2]
]

url = "http://127.0.0.1:8000/predict"  # Ensure server is running

predictions = []
headers = {"Content-Type": "application/json"}

for record in data:
    payload = json.dumps({
        "sepal_length": record[0],
        "sepal_width": record[1],
        "petal_length": record[2],
        "petal_width": record[3]
    })

    response = requests.post(url, data=payload, headers=headers)
    predictions.append(response.json())  # Store full response JSON

print(predictions)
