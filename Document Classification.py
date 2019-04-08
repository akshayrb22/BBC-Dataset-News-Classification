import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textblob import Word

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

df = pd.read_csv('dataset/dataset.csv')
category_id_df = df[['category', 'category_id'
                     ]].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)

df.groupby('type').count().plot(kind='bar')


tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='latin-1',
    ngram_range=(1, 2),
    stop_words='english')

X = tfidf.fit_transform(df.news)
y = df.type

y = df['type'].tolist()
Y = np.array(y)

print("no of features extracted:", X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42)

print("train size:", X_train.shape)
print("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=150, n_jobs=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", c_mat)
print("\nKappa: ", kappa)
print("\nAccuracy: ", acc)
