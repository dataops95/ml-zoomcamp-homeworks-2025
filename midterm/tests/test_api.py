import requests
import json

# 1. Define the endpoint
url = 'http://localhost:5000/predict_api'

# 2. Define the payload (Notice the "data" wrapper)
payload = {
    "data": {
        "MedInc": 5,
        "HouseAge": 30,
        "AveRooms": 6,
        "AveBedrms": 1,
        "Population": 500,
        "AveOccup": 3,
        "Latitude": 34.05,
        "Longitude": -118.25
    }
}

# 3. Send the POST request
print("Sending request...")
response = requests.post(url, json=payload)

# 4. Print results
print(f"Status Code: {response.status_code}")
print("Response JSON:")
print(json.dumps(response.json(), indent=4))
