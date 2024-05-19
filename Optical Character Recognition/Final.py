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
        self.root.geometry("950x500")

        self.translated_speech_path = "translated_speech.mp3"
        self.playing = False  # Flag to indicate if speech is playing

        # Load your CSV data
        self.df = pd.read_csv('Language Detection1.csv')

        # Define a function to remove punctuation and clean text
        def remove_punctuation(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, "")
            text = text.lower()
            return text

        # Clean the 'Text' column
        self.df['Text'] = self.df['Text'].apply(remove_punctuation)

        # Split data into training and testing sets
        X = self.df.iloc[:, 0]
        Y = self.df.iloc[:, 1]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

        # TF-IDF Vectorization
        self.vec = TfidfVectorizer(ngram_range=(2, 3), analyzer='char')

        # Create a pipeline with a logistic regression classifier
        self.model_pipe = Pipeline([('vec', self.vec), ('clf', LogisticRegression())])

        # Train the model
        self.model_pipe.fit(X_train, Y_train)
        
        self.qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

        self.create_ui()

    def create_ui(self):
        # Create the main frame for the entire GUI
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a left frame for buttons and a right frame for result boxes
        self.left_frame = ttk.Frame(self.main_frame, padding=10)
        self.left_frame.grid(row=0, column=0, sticky="ns")

        self.right_frame = ttk.Frame(self.main_frame, padding=10)
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Left Frame (Buttons)
        self.label = ttk.Label(self.left_frame, text="Select an image:")
        self.label.grid(row=0, column=0, columnspan=2)

        self.select_button = ttk.Button(self.left_frame, text="Select Image", command=self.load_image)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=5)

        self.language_label = ttk.Label(self.left_frame, text="Select Target Language:")
        self.language_label.grid(row=2, column=0, columnspan=2)

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
        self.language_dropdown = ttk.Combobox(self.left_frame, textvariable=self.language_var, values=list(indian_languages.keys()))
        self.language_dropdown.grid(row=3, column=0, columnspan=2, pady=5)

        self.translate_button = ttk.Button(self.left_frame, text="Translate", command=self.translate_text)
        self.translate_button.grid(row=4, column=0, columnspan=2, pady=5)

        self.summarize_button = ttk.Button(self.left_frame, text="Summarize", command=self.summarize_text)
        self.summarize_button.grid(row=5, column=0, columnspan=2, pady=5)

        self.speech_label = ttk.Label(self.left_frame, text="Extracted text speech")
        self.speech_label.grid(row=6, column=0, columnspan=2)
        self.play_button = ttk.Button(self.left_frame, text="Play", command=self.play_translated_speech)
        self.play_button.grid(row=7, column=0, pady=5)

        self.pause_button = ttk.Button(self.left_frame, text="Pause", command=self.pause_translated_speech)
        self.pause_button.grid(row=7, column=1, pady=5)

        # Right Frame (Result Boxes)
        self.result_text_frame = ttk.Frame(self.right_frame)
        self.result_text_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.extracted_text_label = ttk.Label(self.result_text_frame, text="Extracted Text:")
        self.extracted_text_label.grid(row=0, column=0, columnspan=2)

        self.extracted_text = tk.Text(self.result_text_frame, wrap=tk.WORD, height=13, width=30)
        self.extracted_text.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.translated_text_label = ttk.Label(self.result_text_frame, text="Translated Text:")
        self.translated_text_label.grid(row=0, column=2, columnspan=2)

        self.translated_text = tk.Text(self.result_text_frame, wrap=tk.WORD, height=13, width=30)
        self.translated_text.grid(row=1, column=2, padx=10, pady=5, sticky="nsew")

        self.summary_text_label = ttk.Label(self.result_text_frame, text="Summarized Text:")
        self.summary_text_label.grid(row=2, column=0, columnspan=2)

        self.summary_text = tk.Text(self.result_text_frame, wrap=tk.WORD, height=10, width=30)
        self.summary_text.grid(row=3, column=0, columnspan=4, padx=10, pady=5, sticky="nsew")

        # Configure grid weights to allow resizing of result boxes
        self.right_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)

        self.result_text_frame.columnconfigure(0, weight=1)
        self.result_text_frame.columnconfigure(1, weight=1)
        self.result_text_frame.columnconfigure(2, weight=1)
        self.result_text_frame.columnconfigure(3, weight=1)
        self.result_text_frame.rowconfigure(1, weight=1)
        self.result_text_frame.rowconfigure(3, weight=1)
        self.qa_frame = ttk.LabelFrame(self.left_frame, text="Question Answering", padding=10)
        self.qa_frame.grid(row=8, column=0, columnspan=2, sticky="nswe")

        self.user_question_label = ttk.Label(self.qa_frame, text="Enter your question:")
        self.user_question_label.grid(row=0, column=0, columnspan=2)

        self.user_question = tk.StringVar()
        self.user_question_entry = ttk.Entry(self.qa_frame, textvariable=self.user_question, width=40)
        self.user_question_entry.grid(row=1, column=0, columnspan=2)

        self.answer_button = ttk.Button(self.qa_frame, text="Get Answer", command=self.answer_question)
        
        self.answer_button.grid(row=2, column=0, columnspan=2, pady=5)

        self.answer_label = ttk.Label(self.qa_frame, text="Answer:")
        self.answer_label.grid(row=3, column=0, columnspan=2)

        self.answer_text = tk.Text(self.qa_frame, wrap=tk.WORD, height=4, width=40)
        self.answer_text.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Configure grid weights for the QA frame
        self.qa_frame.columnconfigure(0, weight=1)
        self.qa_frame.rowconfigure(4, weight=1)

        

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.process_image(file_path)

    def process_image(self, image_path):
        pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        image = Image.open(image_path)
        text = pt.image_to_string(image, lang='eng+hin+mar+mal')
        self.extracted_text.delete(1.0, tk.END)
        self.extracted_text.insert(tk.END, text)

        input_text = self.extracted_text.get(1.0, tk.END)
        # Detect the language of the extracted text
        detected_language = self.detect_language(input_text)

        # Display the detected language to the left of the "Extracted Text" label
        self.extracted_text_label.config(text=f"Extracted Text ({detected_language}):")

    def translate_text(self):
        input_text = self.extracted_text.get(1.0, tk.END)
        target_language = self.language_var.get()

        translator = Translator()
        translated = translator.translate(input_text, dest=target_language)
        translated_text = translated.text

        self.translated_text.delete(1.0, tk.END)
        self.translated_text.insert(tk.END, translated_text)

    def summarize_text(self):
        input_text = self.extracted_text.get(1.0, tk.END)
        nlp = spacy.load('en_core_web_sm')

        doc = nlp(input_text)

        tokens = [token.text for token in doc]

        punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""

        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in STOP_WORDS:
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1

        max_frequencies = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequencies

        sentence_tokens = [sent for sent in doc.sents]

        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]

        if len(input_text) < 1000:
            select_length = int(len(sentence_tokens) * 0.5)
        elif 1000 < len(input_text) < 2000:
            select_length = int(len(sentence_tokens) * 0.4)
        else:
            select_length = int(len(sentence_tokens) * 0.3)

        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)

        final_summary = [word.text for word in summary]

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, '\n'.join(final_summary))

    def play_translated_speech(self):
        if not self.playing:
            pygame.mixer.init()
            pygame.mixer.music.load(self.translated_speech_path)
            pygame.mixer.music.play()
            self.playing = True
            self.root.after(500, self.check_playback_status)

    def pause_translated_speech(self):
        if self.playing:
            pygame.mixer.music.pause()
            self.playing = False

    def check_playback_status(self):
        if not pygame.mixer.music.get_busy():
            self.playing = False

    def detect_language(self, text):
        prediction = self.model_pipe.predict([text])[0]
        return prediction
    
    
  
    def answer_question(self):
                    # Get the user's question and extracted text
                question = self.user_question.get()
                context = self.extracted_text.get(1.0, tk.END).strip()  # Remove leading/trailing whitespace
        
            # Check if the question is empty
                if not question:
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(tk.END, "Please enter a question.")
                    return
        
            # Check if the context (extracted text) is empty
                if not context:
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(tk.END, "Please load an image and extract text first.")
                    return
        
                try:
                    # Import necessary libraries for BERT-based question answering
                    import torch
                    from transformers import BertTokenizer, BertForQuestionAnswering
        
                    # Load the BERT model and tokenizer
                    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
                    model = BertForQuestionAnswering.from_pretrained(model_name)
                    tokenizer = BertTokenizer.from_pretrained(model_name)
        
                    # Tokenize the question and context
                    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True)
        
                    # Get the answer
                    with torch.no_grad():
                            start, end = model(**inputs).values()
        
                    answer_start = torch.argmax(start)
                    answer_end = torch.argmax(end)
        
                    answer = tokenizer.decode(inputs['input_ids'][0][answer_start:answer_end + 1])
        
                    # Display the answer in the GUI

                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(tk.END, answer)
                    # Display the answer in the GUI
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(tk.END, answer)
                except Exception as e:
                    self.answer_text.delete(1.0, tk.END)
                    self.answer_text.insert(tk.END, "An error occurred while answering the question.")
                finally:
                    pass 





        
                    

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRTranslatorApp(root)
    root.mainloop()
    