# Heart Disease Prediction Model

Машинное обучение для предсказания риска сердечных заболеваний на основе медицинских показателей.

## Описание проблемы

Сердечно-сосудистые заболевания - ведущая причина смерти в мире. Ранняя диагностика позволяет:
- Снизить смертность на 30%
- Сократить затраты на лечение
- Улучшить качество жизни пациентов

Модель анализирует 13 медицинских показателей и предсказывает вероятность заболевания.

## Датасет

- **Источник**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Размер**: 920 пациентов
- **Признаки**: 13 медицинских показателей
- **Целевая переменная**: наличие/отсутствие заболевания

## Установка

### Локально
```bash
# Клонировать репозиторий
git clone <repo-url>
cd heart-disease-prediction

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Docker
```bash
# Собрать образ
docker build -t heart-disease-api .

# Запустить контейнер
docker run -p 9696:9696 heart-disease-api
```

## Использование

### Обучение модели
```bash
python train.py
```

### Запуск API
```bash
python serve.py
```

### Пример запроса
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

Ответ:
```json
{
  "prediction": 1,
  "risk": "High",
  "probability": 0.87
}
```

## Результаты

| Модель | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.83 | 0.88 | 0.85 |
| Random Forest | 0.88 | 0.86 | 0.91 | 0.88 |
| XGBoost | **0.90** | **0.89** | **0.92** | **0.90** |

Лучшая модель: **XGBoost** с accuracy 90%

## Структура проекта
```
heart-disease-prediction/
├── data/
│   └── heart_disease.csv
├── models/
│   ├── heart_disease_model.pkl
│   └── scaler.pkl
├── notebooks/
│   └── notebook.ipynb
├── train.py
├── predict.py
├── serve.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Авторы

[Your Name]

## Лицензия

MIT