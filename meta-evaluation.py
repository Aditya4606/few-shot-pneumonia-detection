from tensorflow.keras.models import load_model

best_model = load_model("best_model.keras")

loss, accuracy = best_model.evaluate()