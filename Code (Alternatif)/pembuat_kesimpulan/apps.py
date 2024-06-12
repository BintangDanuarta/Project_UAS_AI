from flask import Flask, request, render_template, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import PyPDF2
from tqdm import tqdm
import time


import pandas as pd
dataset = pd.read_csv('dataset.csv')
print(dataset.head())

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def read_pdf(file):
    """
    Read text from a PDF file.
    """
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def summarize_text(text, update_progress):
    """
    Summarize the input text using T5 model.
    """
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    update_progress(100)
    return summary

progress = 0

@app.route('/progress')
def get_progress():
    global progress
    return jsonify(progress=progress)

def update_progress(value):
    global progress
    progress = value

@app.route('/', methods=['GET', 'POST'])
def index():
    global progress
    summary = ""
    progress = 0
    if request.method == 'POST':
        file = request.files.get('file')
        text = request.form.get('text')
        if file:
            text = read_pdf(file)
            update_progress(50)  # Update progress to 50% after reading PDF
            time.sleep(1)  # Simulate delay
            summary = summarize_text(text, update_progress)
        elif text:
            update_progress(50)  # Update progress to 50% after receiving text
            time.sleep(1)  # Simulate delay
            summary = summarize_text(text, update_progress)
    return render_template('index.html', summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
