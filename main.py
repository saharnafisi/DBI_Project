import csv
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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
    #delete punctuation marks from tokens
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
    for comment in tokenized_comments_no_stopwords:
        new_term_vector = []
        for word in comment:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        tokenized_comments_no_stopwords.append(new_term_vector)

    return tokenized_comments_no_stopwords


if __name__ == "__main__":
    comments = []
    tokenized_comments = []
    tokenized_comments_no_punctuation=[]
    tokenized_comments_no_stopwords=[]
    comments = readFromFile()
    tokenized_comments = tokenize_comments(comments)
    tokenized_comments_no_punctuation=delete_punctuation(tokenized_comments)
    tokenized_comments_no_stopwords=delete_stopwords(tokenized_comments_no_punctuation)
    print(tokenized_comments_no_punctuation)
