FROM python:3.12.3

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

EXPOSE 4014

CMD ["python3", "./app.py"]
