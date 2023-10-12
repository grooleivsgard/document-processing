import random
import codecs
import os
import nltk
import gensim
import logging
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from gensim import models
from gensim import similarities

# 1.0 - random number generator
random.seed(123)


# stopwords = nltk.corpus.stopwords.words('english')
# nltk.download('stopwords')
# nltk.download('punkt')

# DELETE ALL FILES -- for testing
def delete_all_files():
    files_in_directory = os.listdir()

    # Filter the list to include only those named 'paragraph_x.txt'
    filtered_files = [file for file in files_in_directory if "paragraph_" in file and file.endswith(".txt")]

    # Delete each file
    for file in filtered_files:
        os.remove(file)

    print(f"Removed: {len(files_in_directory)}")


# ----- PART 1: Data loading and preprocessing ------

# ---- Partition document collection

def partition(input_file):
    # 1.1 - open and load utf-8 encoded file
    with codecs.open(input_file, "r", "utf-8") as file:
        lines = file.readlines()

        chunks = []
        paragraph = []
        # 1.2 - Partition file into seperate paragraphs. Each paragraph will be a seperate document.
        for line in lines:
            line = line.strip()
            if line:  # if the line is not empty, add to current paragraph
                paragraph.append(line)
            else:  # if the line is empty and there's an existing paragraph, end the current paragraph
                if paragraph:
                    chunks.append(' '.join(paragraph))
                    paragraph = []
        # Add the last paragraph if the file doesn't end with an empty line
        if paragraph:
            chunks.append(' '.join(paragraph))

    for index, paragraph in enumerate(chunks, start=1):
        output_filename = f"paragraph_{index}.txt"
        with codecs.open(output_filename, 'w', "utf-8") as out_file:
            out_file.write(paragraph)

    print(f"Partitioned {len(chunks)} into seperate files.")


def preprocess_collection(target_word, directory_path):
    files_in_directory = os.listdir()
    collection = [f for f in files_in_directory if
                  os.path.isfile(os.path.join(directory_path, f)) and f.startswith("paragraph_") and f.endswith(
                      ".txt")]

    # 1.3 - Remove paragraph (document) if it contains target word
    for file in collection:
        with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
            content = f.read()
            if target_word in content:
                os.remove(os.path.join(directory_path, file))

    return collection


# ---- Tokenize document collection
def tokenize(collection):
    processed_files = []

    # Initialize FreqDist
    fdist = FreqDist()

    # Read files
    for file in collection:
        tokenized_file = tokenize_doc(file)

        # 1.7 - Add processed tokens to list and update fdist
        fdist.update(tokenized_file)
        processed_files.append(tokenized_file)

    return processed_files


def tokenize_doc(file):
    # 4.1 - Apply transformations to query
    punctuation = [',', '.', ';', ':', '?', '!', '(', ')', '[', ']', '{', '}', '"', "'", "’"]

    with open(file, 'r', encoding='utf-8') as f:
        # 1.5 - Convert to lower case
        content = f.read().lower()

        # 1.4 - Tokenize words
        tokens = nltk.word_tokenize(content)

        # 1.6 - Stem tokens and remove punctuation (1.5)
        stemmer = nltk.stem.PorterStemmer()
        stemmed_file = [stemmer.stem(word) for word in tokens if
                        word not in punctuation]
        tokenized_doc = stemmed_file

        return tokenized_doc


# ----- PART 2: Dictionary building ------
def to_bow(processed_files, dictionary):
    # 2.2 - Map paragraphs into Bags-of-Words
    bow_corpus = [dictionary.doc2bow(token, allow_update=True) for token in processed_files]
    return bow_corpus


def build_dictionary(processed_files):
    # 2.1 - Build the dictionary
    dictionary = gensim.corpora.Dictionary(processed_files)

    # Retrieve and filter stopwords
    stopwords = retrieve_stopwords()
    stop_ids = []
    removed = 0
    for word in stopwords:
        try:
            stop_id = dictionary.token2id[word]
            stop_ids.append(stop_id)
            removed += 1
        except:
            pass
    dictionary.filter_tokens(stop_ids)

    return dictionary


def retrieve_stopwords():
    # File downloaded from TextFixer: https://www.textfixer.com/tutorials/common-english-words.php
    file = open('./common-english-words.txt', 'r')
    stopwords = []
    for rows in file:
        row = rows.rstrip().split(',')
        stopwords += row
    return stopwords


# ----- PART 3: Retrieval models ------

def tfidf_conversion(processed_files, dictionary):
    corpus = to_bow(processed_files, dictionary)
    # 3.1 - Initialize TD-IDF model using Bag-of-Words
    tfidf_model = gensim.models.TfidfModel(corpus, normalize=True)
    # 3.2 - Map bow into TF-IDF weights
    tfidf_corpus = tfidf_model[corpus]

    return tfidf_corpus


def retrieve_lsi(dictionary, tfidf_corpus):
    # 3.4 - Initialize LSI model using the TD-IDF corpus
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)  # Create LSI Model
    lsi_corpus = lsi_model[tfidf_corpus]

    # 3.5 - Printing first 3 LSI topics
    # result = lsi_model.show_topics(3)
    # print(result)

    return lsi_corpus


# ----- PART 4: Querying ------

def similarity(query, collection):
    # 2.1 - Build dictionary
    dictionary = build_dictionary(collection)

    # 2.2 - Map paragraphs to Bag-of-Words
    corpus = to_bow(collection, dictionary)

    # -- TD-IDF conversion --
    # 3.1 - Initialize TD-IDF model using Bag-of-Words
    tfidf_model = gensim.models.TfidfModel(corpus, normalize=True)
    # 3.2 - Map bow into TF-IDF weights
    tfidf_corpus = tfidf_model[corpus]

    # --- LSI model ---
    # 3.4 - Initialize LSI model using the TD-IDF corpus
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)  # Create LSI Model
    lsi_corpus = lsi_model[tfidf_corpus]

    # 4.2 - Convert the query to LSI space
    query_vec = dictionary.doc2bow(query)
    lsi_vec = lsi_model[query_vec]

    # ---- Model similarity ---
    # 3.3 - Construct MatrixSimilarity object of the LSI corpus
    index = gensim.similarities.MatrixSimilarity(lsi_corpus)

    # Perform similarity query against the corpus
    sims = index[lsi_vec]
    sims = sorted(enumerate(sims), key=lambda kv: -kv[1])[:3]

    # Report top 3 most relevant paragraphs for the query
    count = 0
    for doc_position, doc_score in sims:
        paragraph = retrieve_paragraph(doc_position)
        print(f"[paragraph {doc_position}] \n {paragraph} \n ")
        count += 1
        if count == 3:
            break


def retrieve_paragraph(doc_position):
    """
    Retrieves the original, unprocessed paragraph.
    """
    directory_path = "./"
    files_in_directory = os.listdir()
    collection = [f for f in files_in_directory if
                  os.path.isfile(os.path.join(directory_path, f)) and f.startswith("paragraph_") and f.endswith(
                      ".txt")]

    for i, file in enumerate(collection):
        if i == doc_position:
            with open(os.path.join(directory_path, file), 'r', encoding='utf-8') as f:
                # Read the first five lines - doesn't work
                content = f.read()
                return content
    return None


def preprocess_query(query):
    """
    Converts a query to a file, and tokenizes file.
    """
    filename = query_to_file(query)
    directory_path = "./"
    if os.path.isfile(os.path.join(directory_path, filename)):
        processed_query = filename
    else:
        processed_query = ''

    return processed_query


def query_to_file(query):
    """
    Writes query to a file and returns the filename query.txt
    """
    filename = "query.txt"
    mode = "w" if os.path.exists(filename) else "x"

    with open(filename, mode) as query_file:
        query_file.write(query)

    return filename


def main():
    filename = "pg3300.txt"
    target_word = "Gutenberg"
    directory_path = "./"
    query1 = "What is the function of money?"
    query2 = "How taxes influence Economics?"
    # processor = DataProcessing(input_filename)

    # ONLY FOR TESTING - Remove all generated files
    # delete_all_files()

    # STEP 1 - Partition all paragraphs into documents
    # partition(input_filename)

    # STEP 2 - Filter collection
    # processed_collection = preprocess_collection(target_word, directory_path)
    # processed_query = preprocess_query(query2)

    # STEP 3 - Tokenize
    # tokenized_collection = tokenize(processed_collection)
    # tokenized_query = tokenize_doc(processed_query)

    # STEP 4 - Run similarity check
    # similarity(tokenized_query, tokenized_collection)


if __name__ == "__main__":
    main()
