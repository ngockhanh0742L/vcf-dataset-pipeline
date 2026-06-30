"""TensorFlow baseline for a processed dataset mounted in a Kaggle notebook."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_rows(data_root, split):
    table = pd.read_csv(data_root / "manifests" / "sequence_manifest.csv")
    table = table[table["split"] == split].copy()
    if table.empty:
        raise ValueError(f"No rows found for split={split}")
    paths = []
    for value in table["frame_paths"]:
        sequence = [str(data_root / "sequences" / item) for item in json.loads(value)]
        paths.append(sequence)
    lengths = {len(item) for item in paths}
    if len(lengths) != 1:
        raise ValueError(f"Split {split} contains inconsistent sequence lengths: {lengths}")
    return np.asarray(paths), table["label"].astype("float32").to_numpy()


def _make_dataset(tf, paths, labels, image_size, batch_size, training):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=42, reshuffle_each_iteration=True)

    def load_sequence(frame_paths, label):
        def load_frame(path):
            image = tf.io.decode_image(
                tf.io.read_file(path), channels=3, expand_animations=False
            )
            image.set_shape([None, None, 3])
            return tf.image.resize(tf.cast(image, tf.float32), [image_size, image_size])

        frames = tf.map_fn(
            load_frame,
            frame_paths,
            fn_output_signature=tf.TensorSpec((image_size, image_size, 3), tf.float32),
        )
        return tf.keras.applications.efficientnet.preprocess_input(frames), label

    dataset = dataset.map(load_sequence, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_model(tf, seq_len, image_size, backbone_name):
    backbones = {
        "b0": tf.keras.applications.EfficientNetB0,
        "b3": tf.keras.applications.EfficientNetB3,
    }
    backbone = backbones[backbone_name](
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input((seq_len, image_size, image_size, 3))
    features = tf.keras.layers.TimeDistributed(backbone)(inputs)
    features = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(features)
    features = tf.keras.layers.Dropout(0.3)(features)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(features)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a TensorFlow sequence baseline")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", default="best_model.keras")
    parser.add_argument("--backbone", choices=["b0", "b3"], default="b0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    import tensorflow as tf

    train_paths, train_labels = _load_rows(args.data_root, "train")
    val_paths, val_labels = _load_rows(args.data_root, "val")
    seq_len = int(train_paths.shape[1])
    sample = tf.io.decode_image(tf.io.read_file(train_paths[0, 0]), channels=3)
    image_size = int(tf.shape(sample)[0].numpy())
    train = _make_dataset(tf, train_paths, train_labels, image_size, args.batch_size, True)
    val = _make_dataset(tf, val_paths, val_labels, image_size, args.batch_size, False)
    model = _build_model(tf, seq_len, image_size, args.backbone)

    counts = np.bincount(train_labels.astype(int), minlength=2)
    class_weight = {
        index: len(train_labels) / (2.0 * count)
        for index, count in enumerate(counts)
        if count
    }
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]
    model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
