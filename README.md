# Arabic Sentiment Analysis

This project is a production-ready Deep Learning API for detecting Arabic sentiment (positive, negative, or neutral). It uses a fine-tuned AraBERT model that has been optimized and exported to ONNX to guarantee highly efficient inference latency. The service is packaged with `pyproject.toml`, tested with `pytest`, and fully containerized via a multi-stage Docker build acting as an automated REST API powered by FastAPI and Pydantic.

## Quickstart: From Zero to Prediction in 3 Commands

You do not need to install Python or clone the entire repository to run the model if the image is published. To build from source and run, execute these exact 3 commands:

```bash
# 1. Build the lightweight, CPU-optimized container image
docker build -f docker/Dockerfile -t arabic-sentiment:latest .

# 2. Run the Docker container exposing Port 8000
docker run -d --rm -p 8000:8000 --name arabic-sentiment-api arabic-sentiment:latest

# 3. Test the deployment by sending an Arabic sentence classification request!
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "المنتج رائع والتوصيل سريع"}'
```

*(You can also visit `http://localhost:8000/docs` in your browser to interact with the interactive Swagger dashboard!)*

## Repository Structure

```text
.
├── data
├── docker
│   └── Dockerfile
├── models
│   └── arabert-twitter-onnx
│       ├── model.onnx
│       ├── model.onnx.data
│       ├── tokenizer_config.json
│       └── tokenizer.json
├── notebooks
│   └── baseline.ipynb
├── pyproject.toml
├── README.md
├── reports
├── src
│   └── arabic_sentiment
│       ├── api
│       │   ├── __init__.py
│       │   ├── main.py
│       │   └── schemas.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── __init__.py
│       ├── metrics.py
│       ├── model.py
│       └── train.py
├── tests
│   ├── conftest.py
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_config.py
│   └── test_metrics.py
└── uv.lock
```
## CI/CD Pipeline & Quality Gates
This project utilizes GitHub Actions to execute a robust pipeline upon every push to [main](cci:1://file:///home/abderraouf/Desktop/mlops%20mena/arabic-sentiment/src/arabic_sentiment/train.py:41:0-152:23). 
It runs `ruff` linting and the `pytest` test suite. Crucially, it features an automated Quality Gate: it reads the generated [reports/metrics.json](cci:7://file:///home/abderraouf/Desktop/mlops%20mena/arabic-sentiment/reports/metrics.json:0:0-0:0) file, and if the newly trained model's `f1_macro` drops below the baseline threshold of `0.6900`, the pipeline forcibly fails with `sys.exit(1)`.
