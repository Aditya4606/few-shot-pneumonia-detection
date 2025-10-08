from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


# dataset directory
train_dir = "data/adult/train"
val_dir = "data/adult/val"
test_dir = "data/adult/test"


# Data Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training"
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)


# Without Optimization
best_model = load_model("best_densenet_model.keras")
train_loss, train_accuracy = best_model.evaluate(train_generator)
test_loss, test_accuracy = best_model.evaluate(test_generator)
report = f"""
Model Performance Report (When layers are frozen)
-----------------------------------------------
Train Loss:      {train_loss:.4f}
Train Accuracy:  {train_accuracy*100:.2f}%
Test Loss:       {test_loss:.4f}
Test Accuracy:   {test_accuracy*100:.2f}%
\n
"""
with open("results/densenet_report.txt", 'a') as f:
    f.write(report)


# With Optimization
best_model = load_model("best_model.keras")
train_loss, train_accuracy = best_model.evaluate(train_generator)
test_loss, test_accuracy = best_model.evaluate(test_generator)
report = f"""
Model Performance Report (When layers are not frozen)
-----------------------------------------------
Train Loss:      {train_loss:.4f}
Train Accuracy:  {train_accuracy*100:.2f}%
Test Loss:       {test_loss:.4f}
Test Accuracy:   {test_accuracy*100:.2f}%
\n
"""
with open("results/densenet_report.txt", 'a') as f:
    f.write(report)
