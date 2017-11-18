import csv
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')


class Comment:

    set_of_words = None

    def __init__(self, id, content, real_class):
        self.id = id
        self.content = content
        self.processed_content = content
        self.tokens = None
        self.real_class = real_class
        self.predicted_class = None

    def __standardize_urls(self):
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.processed_content = re.sub(
            url_regex, "replacedurl", self.processed_content)

    def __remove_special_chars(self):
        self.processed_content = self.processed_content.replace(u'\ufeff', '')

    def __tokenize_comments(self):
        # convert each comment content to tokens
        self.processed_content = word_tokenize(self.processed_content)

    def __delete_punctuation(self):
        # delete punctuation marks from tokens
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        new_review = []

        for token in self.processed_content:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        self.processed_content = new_review

    def __delete_stopwords(self):
        new_review = []

        for token in self.processed_content:
            if not token in stopwords.words('english'):
                new_review.append(token)

        self.processed_content = new_review

    def __stemm_tokens(self):
        stemmed_comments = []
        ps = PorterStemmer()
        new_review = []
        for token in self.processed_content:
            new_review.append(ps.stem(token))

        self.processed_content = new_review

    def preprocess_content(self):
        self.__standardize_urls()
        self.__remove_special_chars()
        self.__tokenize_comments()
        self.__delete_punctuation()
        self.__delete_stopwords()
        self.__stemm_tokens()

        def claculate_distance():
            print("this function must be complete later!")


def read_from_file():
    comments = []
    with open('./dataset/Youtube01-Psy.csv', encoding="utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            comments.append(Comment(row[0], row[3], row[4]))

    return comments


def create_set_of_words(tokenized_comments):
    set_of_words = set()
    for comment in tokenized_comments:
        for token in comment:
            set_of_words.add(token)

    return list(set_of_words)


def save_structured_data_set(set_of_words, tokenized_comments):
    with open('temp.csv', 'w', encoding="utf8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(set_of_words)
        for comment in tokenized_comments:
            new_row = []
            for word in set_of_words:
                new_row.append(comment.count(word))
            writer.writerow(new_row)


if __name__ == "__main__":
    comments = read_from_file()

    for comment in comments:
        comment.preprocess_content()

    tokenized_comments_list = [
        comment.processed_content for comment in comments]

    Comment.set_of_words = create_set_of_words(tokenized_comments_list)
    save_structured_data_set(Comment.set_of_words, tokenized_comments_list)
