import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # or "tensorflow" or "torch"

import keras
import keras_nlp
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
import tensorflow as tf
# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")
MIN_STRING_LEN = 12  # Strings shorter than this will be discarded
BATCH_SIZE = 16

def main():
    preprocessor = keras_nlp.models.GPT2CausalLMPreprocessor.from_preset(
        "gpt2_base_en",
        sequence_length=128,
    )
    gpt2_lm = keras_nlp.models.GPT2CausalLM.from_preset(
        "gpt2_base_en", preprocessor=preprocessor
    )

    raw_train_ds = (
        tf_data.TextLineDataset("./docs/utazas_a_holdra.txt")
        .filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN)
        .batch(BATCH_SIZE)
        .shuffle(buffer_size=256)
    )
    # raw_train_ds = raw_train_ds.take(500)
    # for text in raw_train_ds:
    #     print(text)

    num_epochs = 5

    # Linearly decaying learning rate.
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        5e-5,
        decay_steps=1,
        end_learning_rate=0.0,
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    gpt2_lm.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=loss,
        weighted_metrics=["accuracy"],
    )

    gpt2_lm.fit(raw_train_ds, epochs=num_epochs)
    gpt2_lm.save('./verne_hun.keras')

    # Note: batched inputs expected so must wrap string in iterable
    print(gpt2_lm.generate("Mi a helyzet?", max_length=200))

if __name__ == '__main__':
    main()