import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("coral_model_mobilenet.h5", compile=False)

IMG_SIZE = 224

# Load test image
img_path = "test.jpg"   # put any image here
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print(f"Bleached Coral ({prediction*100:.2f}%)")
else:
    print(f"Healthy Coral ({(1-prediction)*100:.2f}%)")