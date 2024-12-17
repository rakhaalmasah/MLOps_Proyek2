FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

COPY ./muhrakhaal_pipeline/serving_model/1734363009 /models/cc-model/1

EXPOSE 8080

CMD ["python", "app.py"]
