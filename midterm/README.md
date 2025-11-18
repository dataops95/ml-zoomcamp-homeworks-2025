# midterm

```
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
pip install numpy pandas scikit-learn matplotlib gunicorn Flask 
pip freeze > requirements_ml_project.txt

docker build -t my-ml-app:6.0 .
docker-compose up -d [--force-recreate]
```

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Make Prediction
```bash
curl -X POST http://localhost:5000/predict_api \
  -H "Content-Type: application/json" \
  -d '{
    "data":{
        "MedInc": [5],
        "HouseAge": [30],
        "AveRooms": [6],
        "AveBedrms": [1],
        "Population": [500],
        "AveOccup": [3],
        "Latitude": [34.05],
        "Longitude": [-118.25]
    }
}'
```
Response
```bash
05],\x0a"Longitude": [-118.25]\x0a}\x0a}';392023eb-6f8a-442b-8580-ed7f3e120049{
  "prediction": 2.535480165533201
}
```

### Postman
```
POST: http://127.0.0.1:5000/predict_api

Raw:
{
    "data":{
        "MedInc": [5],
        "HouseAge": [30],
        "AveRooms": [6],
        "AveBedrms": [1],
        "Population": [500],
        "AveOccup": [3],
        "Latitude": [34.05],
        "Longitude": [-118.25]
    }
}
```

Output 
```json
{
    "prediction": 2.535480165533201
}
```
