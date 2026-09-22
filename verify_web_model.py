import numpy as np
import tensorflow as tf

from Hydromatic_Simulator.model.model import GeneratorModel
from Hydromatic_Simulator.utils.config import config


# ============================================================
# SETTINGS
# ============================================================

NUM_COORD = config["num-coord"]
STRUCTURE_DIM = config["structure-dim"]
WEIGHTS_DIR = "./Hydromatic_Simulator/model/weights"


# Use a fixed test design.
# IMPORTANT: replace this with a design that you have already
# successfully predicted with the desktop application if possible.
TEST_CODE = (
    "10101010101"
    "00000"
    "1100110011"
    "00000"
    "111111"
    "000000000000000000000000000000000"
)

assert len(TEST_CODE) == STRUCTURE_DIM, (
    f"Expected {STRUCTURE_DIM} bits, got {len(TEST_CODE)}"
)


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    models = []

    for i in range(NUM_COORD):

        model = GeneratorModel()

        dummy_design = tf.zeros(
            (1, STRUCTURE_DIM),
            dtype=tf.float32,
        )

        dummy_position = tf.zeros(
            (1, config["nodal-dim"]),
            dtype=tf.float32,
        )

        # Build model before loading weights.
        _ = model.recursive_generate(
            dummy_design,
            dummy_position,
            training=False,
        )

        encoder_path = (
            f"{WEIGHTS_DIR}/"
            f"encoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        decoder_path = (
            f"{WEIGHTS_DIR}/"
            f"decoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        model.encoder.load_weights(encoder_path)
        model.decoder.load_weights(decoder_path)

        models.append(model)

    return models


# ============================================================
# INFERENCE
# ============================================================

def run_test_inference(models, binary_string):

    design_input = np.asarray(
        [int(x) for x in binary_string],
        dtype=np.float32,
    )

    predictions = []

    for i, model in enumerate(models):

        design_tensor = tf.convert_to_tensor(
            design_input[None, :],
            dtype=tf.float32,
        )

        initial_position = tf.convert_to_tensor(
            config["init-pos"][i][None, :],
            dtype=tf.float32,
        )

        prediction = model.recursive_generate(
            design_tensor,
            initial_position,
            training=False,
        )

        prediction = prediction.numpy()

        # (1, 9, 2) → (9, 2)
        prediction = np.squeeze(
            prediction,
            axis=0,
        )

        predictions.append(prediction)

    return np.stack(
        predictions,
        axis=0,
    )


# ============================================================
# RUN
# ============================================================

print("=" * 60)
print("HYDROMATIC MODEL VERIFICATION")
print("=" * 60)

print(f"Structure dimension : {STRUCTURE_DIM}")
print(f"Number of models    : {NUM_COORD}")
print(f"Test code           : {TEST_CODE}")
print()

print("Loading models...")

models = load_models()

print("✓ All models loaded")
print()

print("Running inference...")

prediction = run_test_inference(
    models,
    TEST_CODE,
)

print("✓ Inference completed")
print()

print("Prediction shape:")
print(prediction.shape)

print()
print("Expected:")
print("(16, 9, 2)")

print()
print("First node:")
print(prediction[0])

print()
print("Last node:")
print(prediction[-1])

print()
print("Maximum absolute value:")
print(np.max(np.abs(prediction)))

print()
print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
