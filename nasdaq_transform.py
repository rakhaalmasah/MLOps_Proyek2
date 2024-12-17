
import tensorflow as tf
import tensorflow_transform as tft

LABEL_KEY = "IXIC"
FEATURE_KEYS = ["AAPL", "MSFT", "AMZN", "BRK_B"]

def preprocessing_fn(inputs):
    outputs = {}
    for key in FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])
    outputs[LABEL_KEY] = inputs[LABEL_KEY]

    return outputs
