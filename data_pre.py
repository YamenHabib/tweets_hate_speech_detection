import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')

from datasets import load_dataset
dataset_dict = load_dataset('tweets_hate_speech_detection', split='train')

data_path = 'data'
df = pd.DataFrame.from_records(dataset_dict)

counts = df['label'].value_counts()
print(f"Count of label 0 (non hate specch) is: {counts[0]} and for label 1 (hate specch) is: {counts[1]}","\n\n")

counts.plot.bar()
tweet_length = df['tweet'].str.len().plot.hist()


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(df.tweet)
sum_words = np.squeeze(np.asarray(words.sum(axis=0)))
words_count = [(word, sum_words[i]) for word, i in cv.vocabulary_.items()]
words_count = sorted(words_count, key = lambda x: x[1], reverse = True)
counts = pd.DataFrame(words_count, columns=['word', 'freq'])
counts.head(20).plot(x='word', y='freq', kind='bar', figsize = (20,9))
plt.title("Most Frequently 20 Word")



# Save preprocessed data, cropped to max length of the model.
df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.split()[:512]))
df.to_csv(f"{data_path}/prep_tweets.csv")
