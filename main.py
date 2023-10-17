import random
import codecs
import os
import re
from nltk.tokenize import RegexpTokenizer
import nltk
import gensim
from nltk.probability import FreqDist
from gensim import models
from gensim import similarities

# 1.0 - random number generator
random.seed(123)


# ----- PART 1: Data loading and preprocessing ------
# ---- Partition document collection
def partition(input_file):
    # 1.1 - open and load utf-8 encoded file
    with codecs.open(input_file, "r", "utf-8") as file:
        lines = file.read()
        # 1.2 - Partition file into seperate paragraphs.
        paragraphs = re.split('\n\s*\n', lines)

    return paragraphs


def preprocess_collection(target_word, paragraphs):
    # 1.3 - Only add paragraphs that does not contain the target word
    filtered_paragraphs = list(filter(lambda x: target_word.lower() not in x.lower(), paragraphs))

    return filtered_paragraphs


# ---- Tokenize document collection
def tokenize_text(text):
    # Initialize FreqDist
    fdist = FreqDist()
    ps = nltk.stem.PorterStemmer()
    stemmed_and_tokenized = []

    # 1.5 - RegExp used to remove punctuation during tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    # 1.5 - Make all paragraphs and target word lowercase
    paragraphs_lowerCase = [s.lower() for s in text]

    # 1.4 - Tokenize paragraphs
    for paragraph in paragraphs_lowerCase:
        tokenized_paragraph = tokenizer.tokenize(paragraph)
        # 1.6 - Stem words
        stemmed_paragraph = [ps.stem(token) for token in tokenized_paragraph]
        stemmed_and_tokenized.append(stemmed_paragraph)
        # 1.7 - Update FreqDist
        fdist.update(stemmed_paragraph)

    return stemmed_and_tokenized


# ----- PART 2: Dictionary building ------

def retrieve_stopwords():
    # File downloaded from TextFixer: https://www.textfixer.com/tutorials/common-english-words.php
    file = open('./common-english-words.txt', 'r')
    stopwords = []
    for rows in file:
        row = rows.rstrip().split(',')
        stopwords += row
    return stopwords


def to_bow(tokenized_paragraphs, dictionary):
    # 2.2 - Map paragraphs into Bags-of-Words
    bow_corpus = []
    for token in tokenized_paragraphs:
        vector = dictionary.doc2bow(token)
        bow_corpus.append(vector)
    # bow_corpus = [dictionary.doc2bow(token, allow_update=True) for token in tokenized_paragraphs]
    return bow_corpus


def build_dictionary(tokenized_paragraphs):
    # 2.1 - Build the dictionary
    dictionary = gensim.corpora.Dictionary(tokenized_paragraphs)

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


# ----- PART 3: Retrieval models ------

def initialize_tfidf(bow_corpus):
    return gensim.models.TfidfModel(bow_corpus, normalize=True)


def initialize_lsi(tfidf_corpus, dictionary):
    return gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)


def construct_matrix_similarity(corpus):
    return gensim.similarities.MatrixSimilarity(corpus)


def get_top_similarities(index, lsi_vec):
    sims = index[lsi_vec]
    return sorted(enumerate(sims), key=lambda kv: -kv[1])[:3]


def similarity(tokenized_query, tokenized_paragraphs, paragraphs):
    # 2.1 - Build dictionary
    dictionary = build_dictionary(tokenized_paragraphs)

    # 2.2 - Map paragraphs to Bag-of-Words
    bow_corpus = to_bow(tokenized_paragraphs, dictionary)

    # -- TD-IDF conversion --
    # 3.1 - Initialize TD-IDF model using Bag-of-Words
    tfidf_model = initialize_tfidf(bow_corpus)
    # 3.2 - Map bow into TF-IDF weights
    tfidf_corpus = tfidf_model[bow_corpus]

    # --- LSI model ---
    # 3.4 - Initialize LSI model using the TD-IDF corpus
    lsi_model = initialize_lsi(tfidf_corpus, dictionary)
    lsi_corpus = lsi_model[tfidf_corpus]

    # ---- Model similarity ---
    # 3.3 - Construct MatrixSimilarity object of the LSI corpus
    tfidf_index = construct_matrix_similarity(tfidf_corpus)
    lsi_index = construct_matrix_similarity(lsi_corpus)

    # 3.5 - Printing first 3 LSI topics
    topics = lsi_model.show_topics(3)
    formatted_topics = []
    print('\n', 'Task 3.5 - Print the first 3 LSI topics.', '\n')
    for topic_id, terms in topics:
        formatted_topic = f"Topic {topic_id}: {terms}"
        formatted_topics.append(formatted_topic)
    print(formatted_topics)

    # 4.2 - Convert the query to LSI space and report TF-IDF weights
    query_vec = dictionary.doc2bow(tokenized_query)
    query_tfidf = tfidf_model[query_vec]

    # Formatting TF-IDF output
    formatted_tfidf = {dictionary[id_]: f"{weight:.2f}" for id_, weight in query_tfidf}
    formatted_tfidf_str = ', '.join([f"{k}: {v}" for k, v in formatted_tfidf.items()])
    print(f"Task 4.2 - TF-IDF weights for query: ({formatted_tfidf_str})")

    # Perform similarity query against the TF-IDF corpus
    sims_tfidf = tfidf_index[query_tfidf]
    doc2sim_tfidf = sorted(enumerate(sims_tfidf), key=lambda kv: -kv[1])[:3]

    # 4.3 - Report top 3 most relevant paragraphs for the query
    print('\n', "Task 4.3 - Report top 3 most relevant paragraphs for the query")
    for doc in doc2sim_tfidf:
        text = paragraphs[doc[0]].split('\n') # Retrieve paragraph from the original, cleaned list
        print('\n', 'Paragraph:', doc[0], '\n')
        if len(text) < 5:
            for i in range(len(text)):
                print(text[i])
        else:
            for i in range(5):
                print(text[i])

    # 4.4
    print('\n', 'Task 4.4a - Report top 3 topics with the most significant (with the largest absolute values) weights')
    lsi_vec = lsi_model[query_tfidf]
    lsi_results = (sorted(lsi_vec, key=lambda kv: -abs(kv[1]))[:3])
    for doc in lsi_results:
        print('\n', 'Topic: ', doc[0], '\n', lsi_model.show_topic(doc[0]))

    sims_lsi = lsi_index[lsi_vec]
    doc2sim_lsi = sorted(enumerate(sims_lsi), key=lambda kv: -kv[1])[:3]

    print('\n', "Task 4.4b - Top 3 most relevant paragraphs according to LSI model", '\n')
    for doc in doc2sim_lsi:
        text = paragraphs[doc[0]].split('\n') # Retrieve paragraph from the original, cleaned list
        print('\n', "Paragraph: ", doc[0], '\n')
        if len(text) < 5:
            for i in range(len(text)):
                print(text[i])
        else:
            for i in range(5):
                print(text[i])


# ----- PART 4: Querying ------

def preprocess_query(query):
    # 4.1 - Preprocess query to remove punctuation, tokenize and stem
    ps = nltk.stem.PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_query = tokenizer.tokenize(query.lower())
    stemmed_query = [ps.stem(token) for token in tokenized_query]

    return stemmed_query


def main():
    filename = "pg3300.txt"
    target_word = "Gutenberg"
    query1 = "What is the function of money?"
    query2 = "How taxes influence Economics?"

    # STEP 1 - Partition all paragraphs into documents
    paragraphs = partition(filename)

    # STEP 2 - Filter collection
    cleaned_paragraphs = preprocess_collection(target_word, paragraphs)

    # STEP 3 - Tokenize and run similarity check
    similarity(preprocess_query(query2), tokenize_text(cleaned_paragraphs), cleaned_paragraphs)


if __name__ == "__main__":
    main()
