import os
import re
import time
import json
import numpy as np
import logging
import dotenv
import os
import pymysql

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from nltk.tokenize import RegexpTokenizer
# Configure the root logger to log INFO and above to logs.txt
logging.basicConfig(filename="logs.txt", level=logging.INFO)

dotenv.load_dotenv()


# Get the MySQL database connection
def get_connection():
    conn = pymysql.connect(
        host=str(os.getenv("SQL_HOST")),
        user=str(os.getenv("SQL_USER")),
        password=str(os.getenv("SQL_PW")),
        database=str(os.getenv("SQL_DB")),
        port=int(os.getenv("SQL_PORT"))
    )
    return conn


# Gets one paper content form a single id
def get_paper_content(id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT distinct content FROM documents_content WHERE document_id = '{id}'")
    result = cursor.fetchone()

    cursor.close()
    conn.close()
    return result


# get all papaer contents form a list of ids
def get_all_paper_contents(ids):
    conn = get_connection()
    cursor = conn.cursor()
    # we need this if statement to prentent writing a the touple as ('id',) which will rise a syntax error
    if len(ids) == 1:
        query = f"""SELECT document_id, content FROM documents_content WHERE document_id = '{ids[0]}' """
    else:
        query = f"SELECT document_id, content FROM documents_content WHERE document_id IN {tuple(ids)}"
    cursor.execute(query)

    result = cursor.fetchall()

    cursor.close()
    conn.close()
    return result


# Performs a curstom querry to the sql database
def query_sql(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""{query}""")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result





def split_long_string(text, chunk_length = 120 , word_lag = 20):

    """
    Splits a single string into an array of strings containing  #chunk_length number of words. 
    Each subsequent string has extra #word_lag words at the begening, being those extra number 
    of words the last words of the previous chunk.
    ----------
    text : str
            A markdown string to be chunked
    chunk_length : int
            The number of words to consider a split
    word_lag : int
            The number of words of the previous chunk to be added at the beginning
            of each subsequent chunk
    """

    words = text.split()
    num_words = len(words)
    segments = []
    start_idx = 0
    end_idx = min(chunk_length, num_words)  # Initial end index for the first 50 words
    while start_idx + word_lag < num_words:
        segment = " ".join(words[start_idx:end_idx])
        segments.append(segment)
        start_idx = end_idx - word_lag  # Move the start index back by 10 words
        end_idx = min(start_idx + chunk_length,
                      num_words)  # Move the end index forward by 40 words or to the end of the text
    return np.array(segments)




def preprocessing(text, min_len = 400, max_len = 4500):



    text = remove_tables_and_pages(text)
    chunks, split_idx = split_string_by_paragraph(text, min_len, max_len)

    return chunks, split_idx




def remove_tables_and_pages(article):


    """
    This Method will split the text by ONLT single line skip  \n (and not \n\n))
    and will delete table and page number indicators. The model will
    fill those deleted lines with lines with spaces "_" equally long 
    as the original line in order to preserve the original lenght of the text so we
    can retrieve strings by their original string idx later.
    """


    lines = re.split(r'\n(?!\n)', article)

    # Find idx to delete
    new_lines = []
    register = False
    for idx,line in enumerate(lines):

        # Delete page start

        if line.startswith("<!-- PAGE"): 
            new_lines.append(" "*len(line))
            continue

        # Delete tables
        if line.startswith("<!--TABLE"):
            new_lines.append(" "*len(line))
            register = True
            continue

        if line.startswith("<!--END"):
            new_lines.append(" "*len(line))
            register = False
            continue

        if register == True:
            new_lines.append(" "*len(line))
            continue

        else:
            new_lines.append(line)


    cleaned_article = '\n'.join(new_lines)




    return cleaned_article





def split_string_by_paragraph(text, min_len = None, max_len = None):

    """
    Splits a single string into an array of strings, the split character is double line skips
    (\n\\n), aka: paragraphs of an article. It filters out paragraphs with less than min_len words, 
    and truncates paragraphs into a maximun lenght of max_len
    """


    matches = [[m.group(0), (m.start(), m.end() - 1)] for m in re.finditer(r'(?!\\n\\n)([^\n]|\\n(?!\\n))+', text)]
    chunked_text, split_idx = zip(*matches)

    # if a min_len is given filters out all the chunks with less than that len
    if min_len != None:
        tup = [(chunk, split_idx[i]) for i,chunk in enumerate(chunked_text) if len(chunk) >= min_len]
        chunked_text = [item[0] for item in tup]
        split_idx = [item[1] for item in tup]

    # if a max_len is given, truncates all the chunks into that lenght


    # Truncate by max_len
    new_chunked_text = []
    new_split_idx = []
    for i, chunk in enumerate(chunked_text):
        length = len(chunk)
        diff = length - max_len

        if diff>0:
            new_chunked_text.append(chunk[0:max_len])
            start, end = split_idx[i][0], split_idx[i][1]
            new_split_idx.append((start, end +1 - diff))
        else: 
            new_chunked_text.append(chunk)
            start, end = split_idx[i][0], split_idx[i][1]
            new_split_idx.append((start, end+1))

    # Assign the new chunked text into what we are returning

    chunked_text = new_chunked_text
    split_idx = new_split_idx

    
    return np.array(chunked_text), np.array(split_idx)


def extract_ngrams_from_article(article_text, n):
    stop_words = set(stopwords.words('english'))
    # Prétraitement
    text = article_text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Supprime tous les caractères non alphanumériques
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words]
    n_grams = ngrams(filtered_words, n)
    n_grams_list = [list(gram) for gram in n_grams]  
    return n_grams_list


# extraire les keywords avec tf-idf
def extract_keywords(article_content):
    tokens = word_tokenize(article_content.lower())
    tokens = [token for token in tokens if token.isalpha()]  # Supprime la ponctuation et les chiffres
    stop_words = set(stopwords.words(
        'english'))  # supprime mots vides like "I, you, he, she, it, we, they,in, on, at, by, with, under, over, between, etc."
    tokens = [token for token in tokens if token not in stop_words]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(tokens)])
    feature_names = vectorizer.get_feature_names_out()
    # scores TF-IDF des mots
    tfidf_scores = tfidf_matrix.toarray().flatten()
    keyword_scores = list(zip(feature_names, tfidf_scores))
    sorted_keywords = sorted(keyword_scores, key=lambda x: x[1], reverse=True)
    # top 5 premiers mots-clés
    return [keyword for keyword, score in sorted_keywords[:5]]
