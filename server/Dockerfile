FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy==1.26.4 scikit-learn==1.4.2 flask gunicorn
COPY server.py .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "server:app"]
