import os

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"

# Note: `userdata.get` is a Colab API. If you're not using Colab, set the env
# vars as appropriate for your system.
os.environ["KAGGLE_USERNAME"] = "gitaroktato"
os.environ["KAGGLE_KEY"] = "ad5730cee5be001b56b201ce97317faf"


import keras
import keras_nlp
# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"

def main():
    gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("code_gemma_2b_en")
    gemma_lm.summary()
    # Note: batched inputs expected so must wrap string in iterable
    # print(gemma_lm.generate("Create hello world in Python", max_length=200))

if __name__ == '__main__':
    main()