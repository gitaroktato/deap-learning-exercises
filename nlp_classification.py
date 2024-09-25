import os

from tensorflow.python.feature_column.utils import sequence_length_from_sparse_tensor

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

import keras
import keras_nlp
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
    # Note: batched inputs expected so must wrap string in iterable
    print(gpt2_lm.generate("Say something!", max_length=200))

if __name__ == '__main__':
    main()