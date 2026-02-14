# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# # Create image generators
# train_gen = ImageDataGenerator(rescale=1./255)
# val_gen = ImageDataGenerator(rescale=1./255)
#
# # Load training data
# train_data = train_gen.flow_from_directory(
#     'dataset/Training',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )
#
# # Load validation data
# val_data = val_gen.flow_from_directory(
#     'dataset/Validation',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='categorical'
# )
#
# print("Classes found:", train_data.class_indices)
# print("Training samples:", train_data.samples)
# print("Validation samples:", val_data.samples)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# -----------------------------
# Data Generators
# -----------------------------
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    'dataset/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    'dataset/Validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# -----------------------------
# CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# -----------------------------
# Save Model
# -----------------------------
model.save("coral_model.h5")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN Model Accuracy')
plt.show()

print("Model training completed and saved as coral_model.h5")
