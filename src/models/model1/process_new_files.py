import fitz

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import os

import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

stops = set(stopwords.words('english'))
stemmer = PorterStemmer()

def get_word_vocab(path):
    vocab = []
    with open(path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]

    return {w: i for i, w in enumerate(vocab)}

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stops]

def build_feature_vector(stems, word2idx, vocab_size=1433):
    vector = [0] * vocab_size
    for s in stems:
        if s in list(word2idx.keys()):
            vector[word2idx[s]] = 1
    return vector

def convert_pdf_to_word_vector(pdf_path, vocab_path):
    word2idx = get_word_vocab(vocab_path)

    pdf_content = extract_text_from_pdf(pdf_path)

    stems = preprocess(pdf_content)

    return np.array(build_feature_vector(stems, word2idx, len(word2idx.keys())))

if __name__ == "__main__":
    vocab_root_folder = r"D:/Sujays documents & files/MS/IDP/Uni Acceptance Letters/DePaul/Classes/Quarter 6/SE489_MLOps/Project/citegraph/src/data/Cora/CoRA_Raw/"

    pdf_root_path = "pdfs/"
    if "pdfs" not in os.listdir():
        os.mkdir("pdfs")
    
    pdf_filename = r"Citation Network.pdf"

    pdf_feature_vector = convert_pdf_to_word_vector(pdf_root_path + pdf_filename, vocab_root_folder + "final_words_dictionary.txt")

    print(pdf_feature_vector)
    print(pdf_feature_vector.shape)