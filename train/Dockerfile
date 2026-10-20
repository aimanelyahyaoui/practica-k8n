FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==1.26.4 scikit-learn==1.4.2
COPY train.py .
CMD ["python", "train.py"]
