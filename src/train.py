import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from src.features import extract_mfcc

DATASET_PATH = "dataset/ravdess"
MODEL_PATH = "models/emotion_model.h5"

X = []
y = []

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Load dataset
for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith('.wav'):
            continue
        file_path = os.path.join(actor_path, file)

        emotion_code = file.split("-")[2]
        emotion = EMOTION_MAP[emotion_code]

        # Original
        mfcc = extract_mfcc(file_path)
        X.append(mfcc)
        y.append(emotion)

        # Augmented version
        mfcc_aug = extract_mfcc(file_path, augment=True)
        X.append(mfcc_aug)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_encoded = to_categorical(y_encoded)

# Reshape for CNN
X = np.expand_dims(X, axis=-1)  # (samples, 40, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ✅ BUILD IMPROVED MODEL (CNN + LSTM)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(40, 1)),
    tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model.summary()

# Train with more epochs and early stopping
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# ✅ SAVE NEW MODEL (NO batch_shape ISSUE)
model.save(MODEL_PATH)

print("✅ New TF 2.10 compatible model saved:", MODEL_PATH)
