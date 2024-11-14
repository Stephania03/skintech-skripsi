from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
import cv2
import joblib
# from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and scaler
model = joblib.load('svm_pso_good_model.pickle')
scaler = joblib.load('scaler.joblib')
# pca = joblib.load('pca.joblib')

def preprocess_image(filepath):
    # Open and resize image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    # Extract HOG features
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
    # Flatten and normalize features
    hog_features = hog_features.reshape(1, -1)
    hog_features_normalized = scaler.transform(hog_features)
    return hog_features_normalized


@app.route('/')
def home():
    return render_template('index.html')

from flask import Flask, render_template, request, redirect
import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model and scaler
model = joblib.load('svm_pso_good_model.pickle')
scaler = joblib.load('scaler.joblib')

def preprocess_image(filepath):
    # Open and resize image
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100))
    # Extract HOG features
    hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
    # Flatten and normalize features
    hog_features = hog_features.reshape(1, -1)
    hog_features_normalized = scaler.transform(hog_features)
    return hog_features_normalized

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'photo' not in request.files:
            return redirect(request.url)
        file = request.files['photo']
        # Check if file is empty
        if file.filename == '':
            return redirect(request.url)
        # Check if file is allowed
        if file:
            # Save uploaded file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Preprocess input data
            input_data = preprocess_image(filepath)
            # Make prediction
            prediction = model.predict(input_data)
            # Delete uploaded file after prediction
            os.remove(filepath)
            # Set threshold
            threshold = 0.1  # You can adjust this threshold as needed
            # Check if prediction probability is above threshold
            if np.max(model.decision_function(input_data)) > threshold:
                # Display prediction
                return render_template('hasil.html', prediction=prediction[0])
            else:
                # If prediction is not above threshold
                return render_template('invalid.html')

if __name__ == '__main__':
    app.run(debug=True)

            
if __name__ == '__main__':
    app.run(debug=True)
