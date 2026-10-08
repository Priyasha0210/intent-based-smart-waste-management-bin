import os
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "gesture_dataset"

COME_FOLDER = os.path.join(
    DATASET_FOLDER,
    "come"
)

NOT_COME_FOLDER = os.path.join(
    DATASET_FOLDER,
    "not_come"
)

MODEL_NAME = "come_gesture_model.keras"

SEQUENCE_LENGTH = 30

FEATURES = 63

RANDOM_STATE = 42

EPOCHS = 100

BATCH_SIZE = 8

# ============================================================
# LOAD DATASET
# ============================================================

print("="*50)
print("Loading Dataset")
print("="*50)

X = []

y = []

# -----------------------------
# COME
# -----------------------------

for file in os.listdir(COME_FOLDER):

    if file.endswith(".npy"):

        path = os.path.join(
            COME_FOLDER,
            file
        )

        sample = np.load(path)

        X.append(sample)

        y.append(1)

# -----------------------------
# NOT COME
# -----------------------------

for file in os.listdir(NOT_COME_FOLDER):

    if file.endswith(".npy"):

        path = os.path.join(
            NOT_COME_FOLDER,
            file
        )

        sample = np.load(path)

        X.append(sample)

        y.append(0)

X = np.array(X,dtype=np.float32)

y = np.array(y,dtype=np.float32)

print("\nDataset Loaded")

print("Total Samples :",len(X))

print("COME :",np.sum(y==1))

print("NOT COME :",np.sum(y==0))

print("Input Shape :",X.shape)

# ============================================================
# SHUFFLE
# ============================================================

indices = np.arange(len(X))

np.random.seed(RANDOM_STATE)

np.random.shuffle(indices)

X = X[indices]

y = y[indices]

# ============================================================
# TRAIN / VALIDATION / TEST
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(

    X,

    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y

)

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,

    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp

)

print("\nTrain :",len(X_train))

print("Validation :",len(X_val))

print("Test :",len(X_test))
# ============================================================
# BUILD IMPROVED LSTM MODEL
# ============================================================

print("\n" + "="*50)
print("Building LSTM Model")
print("="*50)

model = Sequential([

    LSTM(
        128,
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, FEATURES)
    ),

    Dropout(0.30),

    LSTM(
        64,
        return_sequences=False
    ),

    Dropout(0.30),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        1,
        activation="sigmoid"
    )

])

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1

)
checkpoint = ModelCheckpoint(
    MODEL_NAME,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)


callbacks = [

    early_stop,

    checkpoint

]

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "="*50)
print("Training Started")
print("="*50)

history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,

        y_val

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=callbacks,

    shuffle=True,

    verbose=1

)

print("\nTraining Finished!")

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading Best Model...")

model = tf.keras.models.load_model(

    MODEL_NAME

)

print("Best Model Loaded Successfully!")
# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "="*50)
print("TESTING MODEL")
print("="*50)

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# ============================================================
# PREDICTIONS
# ============================================================

pred_prob = model.predict(X_test)

pred = (pred_prob > 0.5).astype(int)

print("\n" + "="*50)
print("TEST PREDICTIONS")
print("="*50)

for i in range(len(pred)):

    actual = "COME" if y_test[i] == 1 else "NOT_COME"

    predicted = "COME" if pred[i][0] == 1 else "NOT_COME"

    print(f"\nSample {i+1}")

    print("Actual     :", actual)

    print("Predicted  :", predicted)

    print("Probability:", round(float(pred_prob[i][0]),3))

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)

print(

classification_report(

    y_test,

    pred,

    target_names=["NOT_COME","COME"]

)

)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    pred

)

print("\nConfusion Matrix")

print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm,cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1],["NOT_COME","COME"])

plt.yticks([0,1],["NOT_COME","COME"])

for i in range(2):

    for j in range(2):

        plt.text(

            j,

            i,

            cm[i,j],

            ha="center",

            va="center",

            color="red",

            fontsize=14

        )

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(

    history.history["accuracy"],

    label="Train Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"

)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(7,5))

plt.plot(

    history.history["loss"],

    label="Train Loss"

)

plt.plot(

    history.history["val_loss"],

    label="Validation Loss"

)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save("come_gesture_model.keras")

print("\n"+"="*50)

print("MODEL SAVED SUCCESSFULLY")

print("File : come_gesture_model.keras")

print("="*50)