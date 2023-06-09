from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from gensim.models.coherencemodel import CoherenceModel
from bertopic.representation import KeyBERTInspired
import gensim.corpora as corpora
# import nltk
# import time

class BERTopicModel():
    def __init__(self):
        # Create custom vectorizer
        self.vectorizer_model = CountVectorizer(stop_words="english")
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.representation_model = KeyBERTInspired()
        self.topic_model = BERTopic(representation_model=self.representation_model, ctfidf_model=self.ctfidf_model, vectorizer_model=self.vectorizer_model, language="indonesian", top_n_words=5, nr_topics="auto")

    def predic_topic(self, tweet_post_cleansed):
        topics, probs = self.topic_model.fit_transform(tweet_post_cleansed)

        return topics, self.topic_model.topic_labels_


