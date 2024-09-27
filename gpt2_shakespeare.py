import os

from tensorflow.python.feature_column.utils import sequence_length_from_sparse_tensor

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras
import keras_nlp
import tensorflow_datasets as tfds
import tensorflow as tf
# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")

def main():
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    shakespeare_ds = tfds.load("tiny_shakespeare", split="train")
    for text in shakespeare_ds:
        print(text['text'])
        break

    train_ds = (
        shakespeare_ds.map(lambda doc: doc['text'])
        .batch(32)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    train_ds = train_ds.take(500)
    num_epochs = 1

    # Linearly decaying learning rate.
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=train_ds.cardinality() * num_epochs,
        end_learning_rate=0.0,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    gpt2_lm.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        weighted_metrics=["accuracy"],
    )

    gpt2_lm.fit(train_ds, epochs=num_epochs)

    # Note: batched inputs expected so must wrap string in iterable
    print(gpt2_lm.generate("Say something!", max_length=200))

if __name__ == '__main__':
    main()