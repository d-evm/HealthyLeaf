# working with the model
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


class_mapping = {
    0: '[APPLE] - Apple Scab',
    1: '[APPLE] - Black Rot',
    2: '[APPLE] - Cedar Apple Rust',
    3: '[APPLE] - HEALTHY',
    4: '[BLUEBERRY] - HEALTHY',
    5: '[CHERRY] - HEALTHY',
    6: '[CHERRY] - Powdery Mildew',
    7: '[CORN] - Cercospora Leaf Spot',
    8: '[CORN] - Common Rust',
    9: '[CORN] - HEALTHY',
    10: '[CORN] - Northern Leaf Blight',
    11: '[GRAPE] - Black Rot',
    12: '[GRAPE] - Black Measles',
    13: '[GRAPE] - HEALTHY',
    14: '[GRAPE] - Leaf Blight',
    15: '[ORANGE] - Citrus Greening',
    16: '[PEACH] - Bacterial Spot',
    17: '[PEACH] - HEALTHY',
    18: '[PEPPER] - Bacterial Spot',
    19: '[PEPPER] - HEALTHY',
    20: '[POTATO] - Early Blight',
    21: '[POTATO] - HEALTHY',
    22: '[POTATO] - Late Blight',
    23: '[RASBERRY] - HEALTHY',
    24: '[SOYABEAN] - HEALTHY',
    25: '[SQUASH] - Powdery Mildew',
    26: '[STRAWBERRY] - HEALTHY',
    27: '[STRAWBERRY] - Leaf Scorch',
    28: '[TOMATO] - Bacterial Spot',
    29: '[TOMATO] - Early Blight',
    30: '[TOMATO] - HEALTHY',
    31: '[TOMATO] - Late Blight',
    32: '[TOMATO] - Leaf Mold',
    33: '[TOMATO] - Septoria Leaf Spot',
    34: '[TOMATO] - Spider Mites',
    35: '[TOMATO] - Target Spot',
    36: '[TOMATO] - Mosaic Virus',
    37: '[TOMATO] - Yellow Leaf Curl Virus'
}


def preprocess_image(path, img_shape=64):
   
    print('Image size: ({})'.format(img_shape))
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)/255
    img = tf.image.resize(img, (img_shape,img_shape))
    img = tf.expand_dims(img, axis=0)
    return img


def predict_disease(image_file):
    preprocessed_image = preprocess_image(image_file)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_disease = class_mapping[predicted_class_index]  
    return predicted_disease

model = load_model('plant_disease_model_2.h5')
image_file = r'dataset\train\Squash___Powdery_mildew\0a079a5d-f0f2-41f5-b2aa-9bbc86203bae___UMD_Powd.M 0307.JPG'
predicted_class_index = predict_disease(image_file)
print("Predicted class index:", predicted_class_index)
