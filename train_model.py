import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------- 1. Basic settings ----------
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

train_dir = "dataset/train"
val_dir = "dataset/val"
test_dir = "dataset/test"

# ---------- 2. Load the datasets from folders ----------
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)

# Improve performance (optional but good)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(100).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ---------- 3. Define a simple CNN model ----------
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=IMG_SIZE + (3,)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPool2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")   # 1 output neuron for binary (good vs defect)
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------- 4. Train the model ----------
EPOCHS = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# ---------- 5. Evaluate on test set ----------
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Test accuracy: {test_acc:.3f}")

# ---------- 6. Save the model ----------
model.save("tile_defect_classifier.h5")
print("\n💾 Model saved as tile_defect_classifier.h5")
