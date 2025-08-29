FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu --default-timeout=1000 -r requirements.txt

COPY . .
# COPY main/.env ./.env

EXPOSE 8000
EXPOSE 5000

# CMD ["uvicorn", "api.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
