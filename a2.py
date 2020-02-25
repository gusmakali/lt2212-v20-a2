import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
from collections import Counter
from nltk.tokenize import word_tokenize
import pandas as pd

from numpy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
random.seed(42)


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)
    return X

#count words lowercased 
def word_counts(lsts, n=5):
    to_df = []
    for count in Counter(lsts).most_common():
        if count[1] < n and count[0].isalpha():
            to_df.append(count)

    return to_df

def tfidf(tf, totaldocs, docswithword):
    return tf * np.log10(totaldocs / (docswithword + 1))

def doc_count(df, word):
    count = 0
    for i in df[word]:
        if i != 0:
            count += 1
    return count

def extract_features(samples):
    print("Extracting features ...")
   
    # raw_text = []
    # for f in samples:
    #     word_nn = word_tokenize(f)
    #     raw_text.append(" ".join(s for s in word_nn if s.isalpha()))
        
    # cv=CountVectorizer()
    # word_count_vector = cv.fit_transform(raw_text)
    # tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    # tfidf_transformer.fit(word_count_vector)

    # count_vector=cv.transform(raw_text)
    # tf_idf_vector=tfidf_transformer.transform(count_vector)
    
    # return tf_idf_vector.toarray()

    column = {}
    wordcounts = {}
    for i, f in enumerate(samples):
        words_and_counts = word_counts(word_tokenize(f.lower()))
    
        if len(words_and_counts) == 0:
            continue
        
        for word in words_and_counts:
            if str(i) not in wordcounts:
                wordcounts[str(i)] = {}
            wordcounts[str(i)][word[0]] = word[1]
            if word[0] not in column:
                column[word[0]] = []
    
    for i, f in enumerate(samples):
        for c in column:
            if str(i) in wordcounts and c in wordcounts[str(i)]:
                column[c].append(wordcounts[str(i)][c])
            else:
                column[c].append(0)
            
    df = pd.DataFrame(column)

    total = len(df)
    tf_idf = df.transform(lambda c: tfidf(c, total, doc_count(df, c.name)))
    array = tf_idf.rename_axis('ID').values
    
    return array
    

# data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
# print(extract_features(data.data[:3]))
# exit()

##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
 
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X_dr[0])
    print("Dim reduced data shape: ", X_dr.shape)
   
    return X_dr


def reduce_dim(X,n):
    #fill this in
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    return svd.transform(X)

#print(part2(extract_features(data.data[:3]), 2))


##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
 
    if clf_id == 1:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf = DecisionTreeClassifier() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))
    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)
    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()
    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)
    # evalute model
    print("Evaluating classcifier ...")
    accuracy, report = evalute_classifier(clf, X_test, y_test)
    print("Accuracy:", accuracy)
    print("Report:", report)



def shuffle_split(X,y):
    X, y = shuffle(X, y) #in case it's not ok
    
    # z = list(zip(X, y))
    # random.shuffle(z)
    # X = list(map(lambda v: list(v[0]), z))
    # y = list(map(lambda v: v[1], z))
    
    X_train = X[:int(len(X)*0.8)]
    X_test = X[int(len(X)*0.8):]
    y_train = y[:int(len(y)*0.8)]
    y_test = y[int(len(y)*0.8):]
    
    return X_train, X_test, y_train, y_test 


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    ## fill in this
    return clf.fit(X, y)


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    #Fill this in
    pred = clf.predict(X)
    return accuracy_score(pred, y), classification_report(pred, y)

######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()

    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples[:1000])

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X_dr = part2(X, n_dim)

    #part 3
    if model_id:
       print("\n------------PART 3-----------")
       part3(X_dr, labels[:1000], model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=args.model_id, 
            n_dim=args.number_dim_reduce
            )
