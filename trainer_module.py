import os
import tensorflow as tf
from tensorflow_transform import TFTransformOutput
from tfx.components.trainer.fn_args_utils import FnArgs

FEATURE_KEYS = ["AAPL", "MSFT", "AMZN", "BRK_B"]
LABEL_KEY = "IXIC"

def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    file_pattern = tf.io.gfile.glob(file_pattern)
    feature_spec = tf_transform_output.raw_feature_spec()

    def parse_fn(serialized_example):
        parsed_features = tf.io.parse_single_example(serialized_example, feature_spec)
        features = tf.concat([tf.cast(parsed_features[key], tf.float32) for key in FEATURE_KEYS], axis=-1)
        label = tf.cast(parsed_features[LABEL_KEY], tf.float32)
        return features, label

    dataset = tf.data.TFRecordDataset(file_pattern, compression_type="GZIP")
    dataset = dataset.map(parse_fn)
    dataset = dataset.shuffle(buffer_size=1000).repeat(num_epochs).batch(batch_size)
    return dataset



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(len(FEATURE_KEYS),), name="input"),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear', name="output"),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def run_fn(fn_args):
    tf_transform_output = TFTransformOutput(fn_args.transform_graph_path)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=fn_args.train_steps)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=1)
    model = build_model()
    model.fit(
        train_dataset,
        validation_data=eval_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        epochs=100
    )
    os.makedirs(fn_args.serving_model_dir, exist_ok=True)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        parsed_features = tf.io.parse_example(serialized_examples, raw_feature_spec)

        features = tf.stack([parsed_features[key] for key in FEATURE_KEYS], axis=1)
        return {"predictions": model(features)}

    tf.saved_model.save(
        model,
        fn_args.serving_model_dir,
        signatures={'serving_default': serve_tf_examples_fn}
    )
