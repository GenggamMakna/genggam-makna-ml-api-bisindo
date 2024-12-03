FROM python:3.12.3

ADD app.py .
ADD model.py .
ADD requirements.txt .

RUN pip install -r requirements.txt

CMD ["python3", "./app.py"]

EXPOSE 8014
