import logging
from flask import Flask, request, jsonify
import tensorflow as tf
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, REGISTRY
import psutil

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

REQUEST_COUNT = Counter('http_request_total', 'Total HTTP Requests', ['method', 'status', 'path'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Duration', ['method', 'status', 'path'])
REQUEST_IN_PROGRESS = Gauge('http_requests_in_progress', 'HTTP Requests in progress', ['method', 'path'])

CPU_USAGE = Gauge('process_cpu_usage', 'Current CPU usage in percent')
MEMORY_USAGE = Gauge('process_memory_usage_bytes', 'Current memory usage in bytes')

MODEL_DIR = "/models/cc-model/1"
loaded_model = tf.saved_model.load(MODEL_DIR)

@app.before_request
def before_request():
    request.start_time = time.time()
    REQUEST_IN_PROGRESS.labels(method=request.method, path=request.path).inc()

@app.after_request
def after_request(response):
    request_latency = time.time() - request.start_time
    REQUEST_COUNT.labels(method=request.method, status=response.status_code, path=request.path).inc()
    REQUEST_LATENCY.labels(method=request.method, status=response.status_code, path=request.path).observe(request_latency)
    REQUEST_IN_PROGRESS.labels(method=request.method, path=request.path).dec()
    return response

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        features = input_data.get('features', {})
        tf_features = {
            key: tf.train.Feature(float_list=tf.train.FloatList(value=value))
            for key, value in features.items()
        }
        example = tf.train.Example(features=tf.train.Features(feature=tf_features))
        serialized_example = example.SerializeToString()
        input_tensor = {'examples': tf.constant([serialized_example])}

        inference_func = loaded_model.signatures['serving_default']
        predictions = inference_func(**input_tensor)
        result = predictions['output_0'].numpy().tolist()

        return jsonify({'predictions': result})
    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': str(e)})

@app.route('/metrics')
def metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.Process().memory_info().rss)
    return generate_latest(REGISTRY), 200, {'Content-Type': CONTENT_TYPE_LATEST}

if __name__ == '__main__':
    logging.info("Starting Flask app on port 8080")
    app.run(debug=True, host="0.0.0.0", port=8080)
