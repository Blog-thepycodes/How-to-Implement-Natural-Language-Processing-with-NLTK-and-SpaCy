import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import scrolledtext, ttk


# We Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# Load SpaCy model
nlp = spacy.load('en_core_web_sm')


# Initialize NLTK's VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()




# Function for text preprocessing using NLTK
def nltk_preprocess(text):
   sentences = sent_tokenize(text)
   words = word_tokenize(text)
   stop_words = set(stopwords.words('english'))


   # Filter out stopwords using filter() and a lambda function
   filtered_words = list(filter(lambda word: word.lower() not in stop_words, words))


   lemmatizer = WordNetLemmatizer()
   lemmatized = [lemmatizer.lemmatize(word) for word in filtered_words]


   return sentences, filtered_words, lemmatized




# Function for text processing using SpaCy
def spacy_process(text):
   doc = nlp(text)
   token_pos_pairs = [(token.text, token.pos_) for token in doc]
   named_entities = [(entity.text, entity.label_) for entity in doc.ents]
   return token_pos_pairs, named_entities




# Function for sentiment analysis using NLTK
def analyze_sentiment(text):
   return sia.polarity_scores(text)




# Function to display results in the GUI
def show_results():
   input_text_content = text_input.get("1.0", tk.END)
   sentences, filtered_words, lemmatized_words = nltk_preprocess(input_text_content)
   token_pos, entities = spacy_process(input_text_content)
   sentiment = analyze_sentiment(input_text_content)


   selected_pos = pos_combobox.get()
   if selected_pos != "All":
       token_pos = [pair for pair in token_pos if pair[1] == selected_pos]


   results = f"Sentences (NLTK):\n{sentences}\n\n"
   results += f"Filtered Words (NLTK):\n{filtered_words}\n\n"
   results += f"Lemmatized Words (NLTK):\n{lemmatized_words}\n\n"
   results += f"Tokens and POS Tags (SpaCy):\n{token_pos}\n\n"
   results += f"Named Entities (SpaCy):\n{entities}\n\n"
   results += f"Sentiment Analysis (NLTK VADER):\n{sentiment}\n"


   text_output.delete("1.0", tk.END)
   text_output.insert(tk.INSERT, results)




# Create the main window
window = tk.Tk()
window.title("Advanced NLP with NLTK and SpaCy - The Pycodes")


# Configure window layout
window.geometry("800x700")
window.grid_columnconfigure(0, weight=1)
window.grid_rowconfigure(0, weight=1)


# Create frames for better layout management
input_frame = ttk.Frame(window, padding="10 10 10 10")
input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


output_frame = ttk.Frame(window, padding="10 10 10 10")
output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


# Create and place widgets in the input frame
label_input = ttk.Label(input_frame, text="Enter Text:")
label_input.grid(row=0, column=0, sticky=tk.W)


text_input = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=70, height=10)
text_input.grid(row=1, column=0, padx=10, pady=10)


pos_label = ttk.Label(input_frame, text="Filter by POS Tag:")
pos_label.grid(row=2, column=0, sticky=tk.W)


pos_combobox = ttk.Combobox(input_frame, values=["All", "NOUN", "VERB", "ADJ", "ADV"], state="readonly")
pos_combobox.set("All")
pos_combobox.grid(row=3, column=0, sticky=tk.W)


process_button = ttk.Button(input_frame, text="Process Text", command=show_results)
process_button.grid(row=4, column=0, pady=10)


# We Create and place widgets in the output frame
label_output = ttk.Label(output_frame, text="Output:")
label_output.grid(row=0, column=0, sticky=tk.W)


text_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, width=70, height=20)
text_output.grid(row=1, column=0, padx=10, pady=10)


# Run the GUI event loop
window.mainloop()
