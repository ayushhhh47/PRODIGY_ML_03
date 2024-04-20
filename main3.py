import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess image data
def load_images(folder_path):
    image_data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, filename))
            img = img.resize((100, 100))  # Resize images to a consistent size
            img_array = np.array(img) / 255.0  # Normalize pixel values to range [0, 1]
            image_data.append(img_array.flatten())
            if 'cat' in filename:
                labels.append(0)  # Assign label 0 for cat
            elif 'dog' in filename:
                labels.append(1)  # Assign label 1 for dog
            else:
                print(f"Warning: Unexpected filename '{filename}', skipping...")
    print("Labels:", labels)  # Debug: Print labels to check distribution
    return np.array(image_data), np.array(labels)


# Path to extracted training and test image folders
train_folder = r'D:\p\pythonProject1\train'  # Update with correct path
test_folder = r'D:\p\pythonProject1\test1'    # Update with correct path

# Load training data
X_train, y_train = load_images(train_folder)

# Load test data (for prediction)
X_test, _ = load_images(test_folder)  # Labels are not needed for test data

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Flatten image data for SVM
X_train_flatten = X_train.reshape(X_train.shape[0], -1)  # Flatten training images
X_test_flatten = X_test.reshape(X_test.shape[0], -1)  # Flatten test images

# Train the SVM classifier
svm_classifier.fit(X_train_flatten, y_train)

# Make predictions on test data
y_pred = svm_classifier.predict(X_test_flatten)

# Create submission file (sampleSubmission.csv format)
submission_data = {'id': np.arange(1, len(y_pred) + 1), 'label': y_pred}
submission_df = pd.DataFrame(submission_data)

# Save submission file
submission_file_path = 'submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file '{submission_file_path}' created successfully!")
