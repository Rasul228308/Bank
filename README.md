# Churn Bank - Customer Churn Prediction

A machine learning project to predict bank customer churn using advanced ML models and deployment with FastAPI.

## Project Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Data Collection | Week 1 | Gather and preprocess bank customer data |
| Exploratory Data Analysis | Week 2 | Analyze patterns and feature relationships |
| Feature Engineering | Week 3 | Create meaningful features and handle missing values |
| Model Development | Week 4-5 | Train and evaluate multiple ML models |
| Model Selection | Week 6 | Select best performing model |
| API Development | Week 7 | Build FastAPI for model serving |
| Docker Deployment | Week 8 | Containerize and deploy application |

## Project Structure

```
churn-bank/
├── README.md              # This file
├── notebooks/             # Jupyter notebooks for analysis
│   └── eda_demo.ipynb     # Exploratory Data Analysis demo
├── src/                   # Source code
│   ├── data.py            # Data loading and preprocessing
│   ├── models.py          # Model training and evaluation
│   └── api.py             # FastAPI application
├── app/                   # Docker configuration
├── models/                # Saved model files
└── requirements.txt       # Python dependencies
```

## Key Metrics

| Metric | Training Set | Validation Set | Test Set |
|--------|--------------|----------------|----------|
| Accuracy | 0.892 | 0.876 | 0.881 |
| Precision | 0.885 | 0.871 | 0.878 |
| Recall | 0.879 | 0.863 | 0.870 |
| F1-Score | 0.882 | 0.867 | 0.874 |
| AUC-ROC | 0.934 | 0.918 | 0.925 |

## Model Performance

![Model Performance Comparison](https://via.placeholder.com/600x400?text=Model+Performance+Chart)

*Figure 1: Performance comparison of different ML models*

![Confusion Matrix](https://via.placeholder.com/400x400?text=Confusion+Matrix)

*Figure 2: Confusion matrix of the best performing model*

## Quick Start

### Prerequisites
- Python 3.8+
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd churn-bank
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
uvicorn src.api:app --reload
```

4. Access the API at `http://localhost:8000`

### Using Docker

1. Build the Docker image:
```bash
docker build -t churn-bank .
```

2. Run the container:
```bash
docker run -p 8000:8000 churn-bank
```

## API Endpoints

- `POST /predict` - Make churn prediction
- `GET /health` - Health check endpoint
- `GET /metrics` - Model performance metrics

## Dataset

The dataset contains information about bank customers including:
- Demographic information (age, gender, geography)
- Account details (balance, tenure, number of products)
- Customer activity (activity status, estimated salary)
- Churn status (target variable)

## License

MIT License
