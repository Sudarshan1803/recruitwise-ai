import os
import re
import nltk
import PyPDF2
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text

def preprocess_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + '\n'
    return preprocess_text(text)

def preprocess_docx(docx_path):
    doc = Document(docx_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return preprocess_text(text)

def preprocess_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == '.pdf':
        return preprocess_pdf(file_path)
    elif extension == '.docx':
        return preprocess_docx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return preprocess_text(text)

def preprocess_resume_directory(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)

        preprocessed_text = preprocess_file(input_file_path)

        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(preprocessed_text)

        print(f'Preprocessed: {filename}')

    print('Preprocessing complete.')