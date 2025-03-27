import pandas as pd

movies = pd.read_csv('movies.csv')

movies = movies[['id', 'title', 'overview', 'genres']]

movies['tags'] = movies['overview'] + movies['genres']

newData = movies.drop(columns=['overview', 'genres'])

from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\\w\\s\\d]', '', text)
    words = wordpunct_tokenize(text)
    stopWords = set(stopwords.words('english'))
    words = [word for word in words if word not in stopWords]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    return text

newData['tags_clean'] = newData['tags'].apply(clean_text)

from sklearn.model_selection import train_test_split

cv = CountVectorizer(max_features=10000, stop_words='english')

vectorizedData = cv.fit_transform(newData['tags_clean'].values.astype('U')).toarray()

vectorizedData.shape

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectorizedData)

newData.info()

distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector: vector[1])

for i in distance[0:5]:
    print(newData.iloc[i[0]].title)

def recommend(movies):
    index = newData[newData['title'] == movies].index[0]

    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])

    for i in distance[0:5]:
        print(newData.iloc[i[0]].title)

recommend("Toy Story")

import pickle

pickle.dump(newData, open('movies_list.pkl', 'wb'))
pickle.dump(newData, open('similaroty.pkl', 'wb'))

pickle.load(open('movies_list.pkl', 'rb'))

import os

print(os.getcwd())