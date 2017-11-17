import csv
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

comments = []


def readFromFile():
    comments = []
    with open('./dataset/Youtube01-Psy.csv', encoding="utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            comments.append(row[3])
            # print(row)
            # print(row[0])
            #print(row[0], row[1], row[2])
    return comments


def tokenize_comments(comments):
    # convert each comment content to tokens
    tokenized_comments = [word_tokenize(comment) for comment in comments]
    return tokenized_comments


def delete_punctuation(tokenized_comments):
    # delete punctuation marks from tokens
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    tokenized_comment_no_punctuation = []

    for review in tokenized_comments:
        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        tokenized_comment_no_punctuation.append(new_review)
    return tokenized_comment_no_punctuation


def delete_stopwords(tokenized_comment_no_punctuation):

    tokenized_comments_no_stopwords = []
    for comment in tokenized_comment_no_punctuation:
        new_term_vector = []
        for word in comment:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        tokenized_comments_no_stopwords.append(new_term_vector)

    return tokenized_comments_no_stopwords


def stemm_tokens(tokenize_comments_no_stemming):
    stemmed_comments = []
    ps = PorterStemmer()
    for comment in tokenize_comments_no_stemming:
        new_term_vector = []
        for word in comment:
            new_term_vector.append(ps.stem(word))
        stemmed_comments.append(new_term_vector)
    return stemmed_comments


if __name__ == "__main__":
    comments = []
    tokenized_comments = []
    tokenized_comments_no_punctuation = []
    tokenized_comments_no_stopwords = []
    stemmed_comments = []
    comments = readFromFile()
    tokenized_comments = tokenize_comments(comments)
    tokenized_comments_no_punctuation = delete_punctuation(tokenized_comments)
    tokenized_comments_no_stopwords = delete_stopwords(
        tokenized_comments_no_punctuation)
    stemmed_comments = stemm_tokens(tokenized_comments_no_stopwords)
    print(stemmed_comments)
