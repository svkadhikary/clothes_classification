import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

# Load your trained model
model = tf.keras.models.load_model('weights.11-0.89-0.36.h5')
# list the categories
categories = ['dress', 'hat', 'longsleeve', 'outwear', 'pants', 'shirt', 'shoes', 'shorts', 'skirt', 't-shirt']

# Set up Flask server
app = Flask(__name__)

# Set up file upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the file from the request
        file = request.files['file']
        # Save the file to the server
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        # Use the model to predict the class of the image
        image = tf.keras.preprocessing.image.load_img(os.path.join('uploads', filename), target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet.preprocess_input(image)
        image = image[tf.newaxis, ...]
        prediction = model.predict(image)
        # Get the class with the highest probability
        class_index = tf.argmax(prediction, axis=1).numpy()[0]
        # Get the class category
        category = categories[class_index]
        # Return the prediction to the client
        return str(category)

# Set up main route
@app.route('/')
def main():
    # Render the file upload form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4444)
