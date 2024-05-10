#using tensorflow
import tensorflow as tf
import numpy as np
import urllib.request
from PIL import Image

# Load pre-trained model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Function to check if image is malicious
def is_malicious(image_path):
    try:
        # Download and open the image
        urllib.request.urlretrieve(image_path, 'image.jpg')
        image = Image.open('image.jpg')
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Predict image class using pre-trained model
        predictions = model.predict(processed_image)
        predicted_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0][1]
        
        # Check if predicted class is malicious
        malicious_classes = ['malware', 'virus', 'trojan', 'adware']  # Add more malicious classes if needed
        if predicted_class.lower() in malicious_classes:
            return True
        
        return False
    
    except Exception as e:
        print("Error:", e)
        return False

# Test the function with an image URL
image_url = 'https://example.com/malicious_image.jpg'
is_image_malicious = is_malicious(image_url)
print("Is image malicious?", is_image_malicious)

#on a large dataset
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Dataset Preparation
csv_path = 'image_dataset.csv'

# Step 2: Data Loading
data = pd.read_csv(csv_path)

# Step 3: Data Preprocessing
# Assuming the 'image_path' column contains the file paths to the images
image_paths = data['image_path'].tolist()
labels = data['label'].tolist()

images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    # Perform any necessary preprocessing steps (e.g., resizing, normalization)
    # and append the preprocessed image to the list
    images.append(image)

# Step 4: Feature Extraction
# Extract features from the preprocessed images using a suitable technique,
# such as a pre-trained CNN or handcrafted features

# Step 5: Model Training
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert image data into flattened feature vectors
X_train = [image.flatten() for image in X_train]
X_test = [image.flatten() for image in X_test]

# Train an SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 7: Prediction
# You can use the trained model to predict the label of a new image
new_image_path = 'path_to_new_image.jpg'
new_image = cv2.imread(new_image_path)
# Perform the same preprocessing steps on the new image as done during training
new_image_preprocessed = preprocess_image(new_image)
new_image_features = new_image_preprocessed.flatten()
prediction = svm_model.predict([new_image_features])
print("Prediction:", prediction)
