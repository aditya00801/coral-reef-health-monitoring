from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create image generators
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

# Load training data
train_data = train_gen.flow_from_directory(
    'dataset/Training',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load validation data
val_data = val_gen.flow_from_directory(
    'dataset/Validation',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

print("Classes found:", train_data.class_indices)
print("Training samples:", train_data.samples)
print("Validation samples:", val_data.samples)
