from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load trained model
model = load_model("coral_model.h5")

# Prepare test data
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    'dataset/Testing',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Predict
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Evaluation
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("Confusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))
