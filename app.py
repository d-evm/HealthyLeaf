from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import io
import csv

model = tf.keras.models.load_model('plant_disease_model_2.h5')

app = Flask(__name__)

disease_name_global = None

@app.route('/')
def welcome_page():
    return render_template('1st_Page.html')

@app.route('/upload_image')
def upload_image():
    return render_template('2nd_Page.html')

@app.route('/predict', methods=['POST'])
def predict():
    global disease_name_global
    def get_disease_name(prediction):
        predicted_class_index = np.argmax(prediction)
        return disease_names.get(predicted_class_index, "Unknown Disease")
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            try:
                processed_image = preprocess_image(image_file.read())
                prediction = model.predict(processed_image)
                disease_name = get_disease_name(prediction)
                disease_name_global = disease_name
                if 'healthy' in disease_name.lower():
                    print("healthy")
                    return render_template('3rd_Page_(1).html')
                return render_template('3rd_Page_(2).html', disease_name=disease_name)
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                return render_template('3rd_Page_(3).html', error_message=error_message)
        else:
            return render_template('2nd_Page.html', error_message="No image uploaded")

@app.route('/4th_Page')
def fourth_page():
    global disease_name_global
    disease_name = disease_name_global

    disease_info, disease_cure = find_disease_info_and_cure(disease_name)

    image_filename = f"{disease_name}.JPG"
    image_path = f"/disease info dataset/{image_filename}" 

    print(disease_name)
    print(disease_info)
    print(disease_cure)

    return render_template('4th_Page.html', disease=disease_name, info=disease_info, cure=disease_cure, image_path=image_path)

def preprocess_image(image_data):
    from PIL import Image
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((64, 64))  
    img = np.array(img)
    img = img / 255.0 
    img = img.astype(np.float32) 
    img = np.expand_dims(img, axis=0)  
    return img

disease_names = {
    0: '[APPLE] - Apple Scab',
    1: '[APPLE] - Black Rot',
    2: '[APPLE] - Cedar Apple Rust',
    3: '[APPLE] - HEALTHY',
    4: '[BLUEBERRY] - HEALTHY',
    5: '[CHERRY] - Powdery Mildew',
    6: '[CHERRY] - HEALTHY',
    7: '[CORN] - Cercospora Leaf Spot',
    8: '[CORN] - Common Rust',
    9: '[CORN] - Northern Leaf Blight',
    10: '[CORN] - HEALTHY',
    11: '[GRAPE] - Black Rot',
    12: '[GRAPE] - Black Measles',
    13: '[GRAPE] - Leaf Blight',
    14: '[GRAPE] - HEALTHY',
    15: '[ORANGE] - Citrus Greening',
    16: '[PEACH] - Bacterial Spot',
    17: '[PEACH] - HEALTHY',
    18: '[PEPPER] - Bacterial Spot',
    19: '[PEPPER] - HEALTHY',
    20: '[POTATO] - Early Blight',
    21: '[POTATO] - Late Blight',
    22: '[POTATO] - HEALHTY',
    23: '[RASBERRY] - HEALTHY',
    24: '[SOYABEAN] - HEALTHY',
    25: '[SQUASH] - Powdery Mildew',
    26: '[STRAWBERRY] - Leaf Scorch',
    27: '[STRAWBERRY] - HEALTHY',
    28: '[TOMATO] - Bacterial Spot',
    29: '[TOMATO] - Early Blight',
    30: '[TOMATO] - Late Blight',
    31: '[TOMATO] - Leaf Mold',
    32: '[TOMATO] - Septoria Leaf Spot',
    33: '[TOMATO] - Spider Mites',
    34: '[TOMATO] - Target Spot',
    35: '[TOMATO] - Yellow Leaf Curl Virus',    
    36: '[TOMATO] - Mosaic Virus',
    37: '[TOMATO] - HEALHTY'
}

disease_descriptions = {
    '[APPLE] - Apple Scab' : 'Description',
    '[APPLE] - Black Rot' : 'Description',
    '[APPLE] - Cedar Apple Rust' : 'Description',
    '[APPLE] - HEALTHY' : 'Description',
    '[BLUEBERRY] - HEALTHY' : 'Description',
    '[CHERRY] - Powdery Mildew' : 'Description',
    '[CHERRY] - HEALTHY' : 'Description',
    '[CORN] - Cercospora Leaf Spot' : 'Description',
    '[CORN] - Common Rust' : 'Description',
    '[CORN] - Northern Leaf Blight' : 'Description',
    '[CORN] - HEALTHY' : 'Description',
    '[GRAPE] - Black Rot' : 'Description',
    '[GRAPE] - Black Measles' : 'Description',
    '[GRAPE] - Leaf Blight' : 'Description',
    '[GRAPE] - HEALTHY' : 'Description',
    '[ORANGE] - Citrus Greening' : 'Description',
    '[PEACH] - Bacterial Spot' : 'Description',
    '[PEACH] - HEALTHY' : 'Description',
    '[PEPPER] - Bacterial Spot' : 'Description',
    '[PEPPER] - HEALTHY' : 'Description',
    '[POTATO] - Early Blight' : 'Description',
    '[POTATO] - Late Blight' : 'Description',
    '[POTATO] - HEALHTY' : 'Description',
    '[RASBERRY] - HEALTHY' : 'Description',
    '[SOYABEAN] - HEALTHY' : 'Description',
    '[SQUASH] - Powdery Mildew' : 'Description',
    '[STRAWBERRY] - Leaf Scorch' : 'Description',
    '[STRAWBERRY] - HEALTHY' : 'Description',
    '[TOMATO] - Bacterial Spot' : 'Description',
    '[TOMATO] - Early Blight' : 'Description',
    '[TOMATO] - Late Blight' : 'Description',
    '[TOMATO] - Leaf Mold' : 'Description',
    '[TOMATO] - Septoria Leaf Spot' : 'Description',
    '[TOMATO] - Spider Mites' : 'Description',
    '[TOMATO] - Target Spot' : 'Description',
    '[TOMATO] - Yellow Leaf Curl Virus' : 'Description',
    '[TOMATO] - Mosaic Virus' : 'Description',
    '[TOMATO] - HEALHTY' : 'Description',
}

def get_disease_name(prediction):
    predicted_class_index = np.argmax(prediction)
    return disease_names.get(predicted_class_index, "Unknown Disease")

def find_disease_info_and_cure(disease_name):
    with open('disease info dataset\data3.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip headers
        for row in reader:
            # print(disease_name.lower())
            # print(row[1].lower())
            # print()

            if row[1].lower() == disease_name.lower():
                disease_info = row[2]
                disease_cure = row[3]
                return disease_info, disease_cure
    return None, None  # Return None if disease not found

if __name__ == '__main__':
    app.run(debug=True)
