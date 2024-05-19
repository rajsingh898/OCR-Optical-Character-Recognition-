import os
import pytesseract as pt
import cv2
from PIL import Image
from googletrans import Translator
from gtts import gTTS
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Load your CSV data
df = pd.read_csv('Language Detection1.csv')

# Define a function to remove punctuation and clean text
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    text = text.lower()
    return text

# Clean the 'Text' column
df['Text'] = df['Text'].apply(remove_punctuation)

# Split data into training and testing sets
X = df.iloc[:, 0]
Y = df.iloc[:, 1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

# TF-IDF Vectorization
vec = TfidfVectorizer(ngram_range=(2, 3), analyzer='char')

# Create a pipeline with a logistic regression classifier
model_pipe = Pipeline([('vec', vec), ('clf', LogisticRegression())])

# Train the model
model_pipe.fit(X_train, Y_train)

# Set the path to the Tesseract executable
pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define functions for language detection and text-to-speech
def detect_language(text):
    prediction = model_pipe.predict([text])[0]
    return prediction

def text_to_speech(text, lang):
    tts = gTTS(text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")

# Path to your test image directory
test_img_path = "/Users/vishn/OCR project/test images/"
create_path = lambda f: os.path.join(test_img_path, f)

test_img_files = os.listdir(test_img_path)
# Create full image path
image_path = test_img_files[7]
path = create_path(image_path)
# Open the image with PIL
image = Image.open(path)

# Image preprocessing
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    # Apply thresholding (adjust the parameters as needed)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal (adjust kernel size as needed)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Contrast enhancement (adjust the parameters as needed)
    enhanced = cv2.equalizeHist(opening)

    # Rescale the image (adjust the size as needed)
    rescaled = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoising (adjust kernel size and sigma as needed)
    denoised = cv2.GaussianBlur(rescaled, (5, 5), 0)

    return Image.fromarray(denoised)

# Perform OCR on the preprocessed image
preprocessed_image = preprocess_image(image)
extracted_text = pt.image_to_string(preprocessed_image, lang='eng')  # Default language is set to English

# Detect the language of the extracted text
detected_lang = detect_language(extracted_text)

# Print the detected language
print(f"Detected Language for {image_path}: {detected_lang}")
