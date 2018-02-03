# database implementation project
# Subject: Comment Spam Filtering on YouTube
# Based on a paper in 'Machine Learning and Applications (ICMLA), 2015 IEEE 14th International Conference'
# Sahar Nafisi      (9321170026)
# Alireza Mohammadi (9321170019)
# Advisor: Dr Ayoub Bagheri
# Kashan University(Winter 2018)

# for reading list of available datasets
import os

# A library for working with dataset files (*.csv)
import csv

# Regular expressions
import re
import string

# natural language toolkit
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')


class Comment:
    """Every comment becomes an instance of this class"""

    # A static variable with dictionary type  that is common between all objects of this class
    # containes a set of all words without duplicat (bag of words)
    # and number of occurance of that word in dataset
    set_of_words = None

    def __init__(self, id, content, real_class):

        # comment ID
        self.id = id

        # raw text of comment (not processed)
        self.content = content

        # processed content of comment (at first this variable is same as 'content')
        self.processed_content = content

        # comment class that is read from dataset (for evaluate result of prediction)
        self.real_class = "spam" if(real_class == "1") else "no_spam"

        # class that our algorithm predict (spam or ham)
        self.predicted_class = None

        # N dimension Vector (N: number of items in 'set_of_words')
        self.vector = []

    def __standardize_urls(self):
        # Replace all of URLs in comment with "replacedurl" phrase
        # this will improve accuracy
        url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        self.processed_content = re.sub(
            url_regex, "replacedurl", self.processed_content)

    def __remove_special_chars(self):
        # in some comments there was some weird phrases that we don't know what are theme
        # probabely these have been created after coping comments from .html pages
        # anyway we removed theme bacause presence of theme reduces accuacy
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
        # remove stopwords (such as 'the, is, at, which, on and ...')
        new_review = []

        for token in self.processed_content:
            if not token in stopwords.words('english'):
                new_review.append(token)

        self.processed_content = new_review

    def __stemm_tokens(self):
        # reducing inflected (or sometimes derived) words to their word stem, base or root form
        stemmed_comments = []
        ps = PorterStemmer()
        new_review = []
        for token in self.processed_content:
            new_review.append(ps.stem(token))

        self.processed_content = new_review

    def preprocess_content(self):
        # functions for preprocessing comment executes continuous in this function
        self.__standardize_urls()
        self.__remove_special_chars()
        self.__tokenize_comments()
        self.__delete_punctuation()
        self.__delete_stopwords()
        self.__stemm_tokens()

    def init_vector(self):
        # initialize to comment vector based on tf-idf of every word in comment

        # Throw an error if set_of_words has None value
        if self.set_of_words is None:
            raise ValueError("run 'create_set_of_words' function first")

        for word in self.set_of_words:
            self.vector.append(self.processed_content.count(word))

    def claculate_similarity(comment):
        # calculate similarity of 2 comments based on 'Cosine Similarity'
        # TODO: complete this function
        print("this function must be complete later!")


def list_of_datesets():
    # returns list of files in 'datasets' directory
    return os.listdir("datasets")


def read_from_file(file_name):
    # reads the dataset file and puts each comment in an 'Comment' object
    # and append all abjects to a list named 'comments' and returns the list
    comments = []
    with open(f'./datasets/{file_name}', encoding="utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            # row[0]: id ,row[3]: content, row[4]: real_class
            comments.append(Comment(row[0], row[3], row[4]))

    return comments


def create_set_of_words(tokenized_comments):
    # Iterates over all of comments and add none repetitive words(tokens) to dictionary
    set_of_words = {}
    for comment in tokenized_comments:
        for token in comment:
            if not token in set_of_words:
                set_of_words[token] = 1  # first occurance of word in data set
            else:
                set_of_words[token] += 1

    Comment.set_of_words = set_of_words


def preprocess_training_dataset(training_dataset):
    # this function creates set of words and saves structured data on disk
    for comment in training_dataset:
        comment.preprocess_content()

    tokenized_comments_list = [
        comment.processed_content for comment in training_dataset]
    Comment.set_of_words = create_set_of_words(tokenized_comments_list)

    for comment in training_dataset:
        comment.init_vector()


def save_structured_data_set(tokenized_comments):
    # TODO: Use vector attribute of Comment class to save data on disk
    with open('temp.csv', 'w', encoding="utf8", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(set_of_words)
        for comment in tokenized_comments:
            new_row = []
            for word in set_of_words:
                new_row.append(comment.count(word))
            writer.writerow(new_row)


if __name__ == "__main__":
    available_datasets = list_of_datesets()

    for i in range(0, len(available_datasets)):
        print(f"{i+1}.{available_datasets[i]}")

    selected_ds_index = int(
        input("Please select one of the above datasets for processing: ")) - 1

    comments = read_from_file(available_datasets[selected_ds_index])

    # 70٪ of selected dataset uses for training and 30% remaining uses for testing
    training_dataset = comments[: int(0.7 * len(comments))]
    test_case_dataset = comments[int(0.7 * len(comments)):]

    # preprocess_training_dataset(training_dataset)
