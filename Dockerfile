# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install C++ build tools
RUN apt-get update && apt-get install -y build-essential && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]