import csv
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')


def readFromFile():
    comments = []
    with open('./dataset/Youtube01-Psy.csv', encoding="utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            comments.append(row[3])
            # print(row)
            # print(row[0])
            # print(row[0], row[1], row[2])

    return comments


def standardize_urls(comments):
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_review = []
    for comment in comments:
        new_review.append(re.sub(url_regex, "replacedurl", comment))

    return new_review


def remove_special_chars(comments):
    new_review = []
    for comment in comments:
        new_review.append(comment.replace(u'\ufeff', ''))

    return new_review


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


def delete_stopwords(tokenized_comments):

    tokenized_comments_no_stopwords = []
    for comment in tokenized_comments:
        new_term_vector = []
        for word in comment:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        tokenized_comments_no_stopwords.append(new_term_vector)

    return tokenized_comments_no_stopwords


def stemm_tokens(tokenized_comments):
    stemmed_comments = []
    ls = LancasterStemmer()
    for comment in tokenized_comments:
        new_term_vector = []
        for word in comment:
            new_term_vector.append(ls.stem(word))
        stemmed_comments.append(new_term_vector)

    return stemmed_comments


def create_set_of_words(tokenized_comments):
    set_of_words = set()
    for comment in tokenized_comments:
        for token in comment:
            set_of_words.add(token)

    return set_of_words


if __name__ == "__main__":
    comments = readFromFile()
    comments = standardize_urls(comments)
    comments = remove_special_chars(comments)
    tokenized_comments = tokenize_comments(comments)
    tokenized_comments = delete_punctuation(tokenized_comments)
    tokenized_comments = delete_stopwords(tokenized_comments)
    tokenized_comments = stemm_tokens(tokenized_comments)
    set_of_words = create_set_of_words(tokenized_comments)
    print(set_of_words)
