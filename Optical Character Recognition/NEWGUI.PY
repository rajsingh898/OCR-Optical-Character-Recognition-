import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image
import pytesseract as pt
from googletrans import Translator
import pygame
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import string
import tkinter as tk
from tkinter import filedialog
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
from transformers import pipeline

class OCRTranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OCR Translator and Summarizer App")
        self.root.geometry("1200x800")
        self.root.configure(bg="#F0F0F0")  # Set a light background color

        self.translated_speech_path = "translated_speech.mp3"
        self.playing = False

        self.df = pd.read_csv('Language Detection1.csv')

        def remove_punctuation(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, "")
            text = text.lower()
            return text

        self.df['Text'] = self.df['Text'].apply(remove_punctuation)

        X = self.df.iloc[:, 0]
        Y = self.df.iloc[:, 1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        self.vec = TfidfVectorizer(ngram_range=(2, 3), analyzer='char')

        self.model_pipe = Pipeline([('vec', self.vec), ('clf', LogisticRegression())])

        self.model_pipe.fit(X_train, Y_train)

        self.qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

        self.create_ui()

    def create_ui(self):
        # Create a main frame to hold the left and right frames
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a left frame for buttons and widgets
        left_frame = ttk.Frame(main_frame, padding=10, relief="raised", borderwidth=2)
        left_frame.grid(row=0, column=0, sticky="ns", padx=10)

        # Create a right frame for result boxes
        right_frame = ttk.Frame(main_frame, padding=10, relief="raised", borderwidth=2)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=10)

        # Left Frame (Buttons)
        label = ttk.Label(left_frame, text="Select an image:", font=("Arial", 14, "bold"))
        label.grid(row=0, column=0, columnspan=2, pady=10)

        select_button = ttk.Button(left_frame, text="Select Image", command=self.load_image, style="Accent.TButton")
        select_button.grid(row=1, column=0, columnspan=2, pady=5)

        language_label = ttk.Label(left_frame, text="Select Target Language:", font=("Arial", 14, "bold"))
        language_label.grid(row=2, column=0, columnspan=2, pady=10)

        indian_languages = {
            "english": "English",
            "hindi": "Hindi",
            "bengali": "Bengali",
            "telugu": "Telugu",
            "marathi": "Marathi",
            "tamil": "Tamil",
            "urdu": "Urdu",
            "gujarati": "Gujarati",
            "kannada": "Kannada",
            "malayalam": "Malayalam",
            "punjabi": "Punjabi",
            "odia": "Odia",
            "assamese": "Assamese",
            "kashmiri": "Kashmiri",
            "nepali": "Nepali",
            "sindhi": "Sindhi",
            "konkani": "Konkani",
            "maithili": "Maithili",
            "bodo": "Bodo",
            "manipuri": "Manipuri"
        }

        self.language_var = tk.StringVar()
        self.language_dropdown = ttk.Combobox(left_frame, textvariable=self.language_var, values=list(indian_languages.keys()), font=("Arial", 12))
        self.language_dropdown.grid(row=3, column=0, columnspan=2, pady=5)

        translate_button = ttk.Button(left_frame, text="Translate", command=self.translate_text, style="Accent.TButton")
        translate_button.grid(row=4, column=0, columnspan=2, pady=5)

        summarize_button = ttk.Button(left_frame, text="Summarize", command=self.summarize_text, style="Accent.TButton")
        summarize_button.grid(row=5, column=0, columnspan=2, pady=5)

        speech_label = ttk.Label(left_frame, text="Extracted text speech", font=("Arial", 14, "bold"))
        speech_label.grid(row=6, column=0, columnspan=2, pady=10)

        play_button = ttk.Button(left_frame, text="Play", command=self.play_translated_speech, style="Accent.TButton")
        play_button.grid(row=7, column=0, pady=5)

        pause_button = ttk.Button(left_frame, text="Pause", command=self.pause_translated_speech, style="Accent.TButton")
        pause_button.grid(row=7, column=1, pady=5)

        # Right Frame (Result Boxes)
        result_text_frame = ttk.Frame(right_frame, relief="groove", borderwidth=2, padding=10)
        result_text_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        extracted_text_label = ttk.Label(result_text_frame, text="Extracted Text:", font=("Arial", 14, "bold"))
        extracted_text_label.grid(row=0, column=0, columnspan=2, pady=5)

        extracted_text = tk.Text(result_text_frame, wrap=tk.WORD, height=13, width=30, font=("Courier", 12))
        extracted_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        translated_text_label = ttk.Label(result_text_frame, text="Translated Text:", font=("Arial", 14, "bold"))
        translated_text_label.grid(row=0, column=2, columnspan=2, pady=5)

        translated_text = tk.Text(result_text_frame, wrap=tk.WORD, height=13, width=30, font=("Courier", 12))
        translated_text.grid(row=1, column=2, padx=10, pady=5, sticky="nsew")

        summary_text_label = ttk.Label(result_text_frame, text="Summarized Text:", font=("Arial", 14, "bold"))
        summary_text_label.grid(row=2, column=0, columnspan=2, pady=5)

        summary_text = tk.Text(result_text_frame, wrap=tk.WORD, height=10, width=30, font=("Courier", 12))
        summary_text.grid(row=3, column=0,