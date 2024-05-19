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
import langid  # Import the language detection function from langid

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

# Set the path to the Tesseract executable
pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define functions for text-to-speech
def text_to_speech(text, lang):
    tts = gTTS(text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")

# Path to your test image directory
test_img_path = "/Users/vishn/OCR project/test images/"
create_path = lambda f: os.path.join(test_img_path, f)

test_img_files = os.listdir(test_img_path)
# Create full image path
image_path = test_img_files[18]
path = create_path(image_path)
# Open the image with PIL
image = Image.open(path)

# Image preprocessing (same as before)


# Perform OCR on the preprocessed image
#preprocessed_image = preprocess_image(image)
extracted_text = pt.image_to_string(image)  # Default language is set to English

# Detect the language of the extracted text using langid
detected_lang, _ = langid.classify(extracted_text)

# Print the detected language
print(f"Detected Language for {image_path}: {detected_lang}")
