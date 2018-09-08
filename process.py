import numpy
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def read_and_clean_training_data(file):
    """
    The following function reads the training data and split them based the label
    :param file:
    :return:
    """
    with open(file, encoding="utf8", errors="ignore") as f:
        stop_words = set(stopwords.words('english'))
        lines = f.readlines()
        x_train = []
        y_train = []
        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(len(lines)):
            line = lines[i]
            if i == 0:
                continue
            else:
                data, label = line.split('\t')
                label = label.strip()
                tokenized_data = tokenizer.tokenize(data)
                cleaned_data = [word.lower() for word in tokenized_data if
                                not word.isdigit() and word != "ml" and word not in stop_words]
                final_data = " ".join(cleaned_data)
                x_train.append(final_data)
                y_train.append(label)
        return x_train, y_train


def train_model(x_train, y_train):
    """
    Using that function you are able to train the Classifier
    :param x_train: train data
    :param y_train: train labels
    :return: classfier object
    """
    clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LinearSVC()),
                    ])

    clf.fit(x_train, y_train)
    return clf


def clean_test_data(test_data, test_labels):
    """
    Using that function you are able to read the test files and make the predictions
    :param test_data: test data file
    :param test_labels: test labels file
    :return:
    """
    xtest = []
    ytest = []
    with open(test_data, encoding="utf8", errors="ignore") as data:
        stop_words = set(stopwords.words('english'))
        tokenizer = RegexpTokenizer(r'\w+')
        lines = data.readlines()
        for i in range(len(lines)):
            line = lines[i]
            if i == 0:
                continue
            else:
                tokenized_line = tokenizer.tokenize(line)
                cleaned_data = [word.lower() for word in tokenized_line if
                                not word.isdigit() and word != "ml" and word not in stop_words]
                xtest.append(" ".join(cleaned_data))
    with open(test_labels, encoding="utf8", errors="ignore") as labels:
        ydata = labels.readlines()
        for label in ydata:
            ytest.append(label.strip())
    return xtest, ytest


def predict(xtest, ytest, clf):
    """
    That function returns the classifiers accuracy
    :param xtest: test data
    :param ytest: test labels
    :param clf: classifier obj
    :return: accuracy
    """
    predicted = clf.predict(xtest)
    return numpy.mean(predicted == ytest)


x_train, y_train = read_and_clean_training_data("training.txt")
clf = train_model(x_train, y_train)

xtest, ytest = clean_test_data("x_test.txt", "y_test.txt")
accuracy = predict(xtest, ytest, clf)
print(accuracy)
