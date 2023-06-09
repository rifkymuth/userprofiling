from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import regex as re
import unidecode
import nltk

# The code for lemmatization
class Lemmatizer():
    def __init__(self):
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()

    def stem(self, text):
        return self.stemmer.stem(text)

class data_preprocessing():
    def __init__(self):
        self.PATH = "TODO"
        nltk.download('wordnet')
        nltk.download('omw-1.4')

        # Create a slang list
        slang_path_file = self.PATH + "slang_id.txt"
        my_file = open(slang_path_file, "r")
        data = my_file.read()
        slang = data.split("\n")
        for i in range(len(slang)):
            slang[i] = slang[i].split(":")
        slangs = dict(slang)
        my_file.close()
        self.slangs = slangs

        # Create a stopwords list
        stopword_path_file = self.PATH + "stopwords_id.txt"
        my_file = open(stopword_path_file, "r")
        data = my_file.read()
        stopwords = data.split("\n")
        my_file.close()
        self.stopwords = stopwords
        
    def remove_newlines_tabs(self, text):    
        # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
        Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')
        return Formatted_text

    def strip_html_tags(self, text):
        # Initiating BeautifulSoup object soup.
        soup = BeautifulSoup(text, "html.parser")
        # Get all the text other than html tags.
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    def remove_links(self, text):
        # Removing all the occurrences of links that starts with https
        remove_https = re.sub(r'http\S+', '', text)
        # Remove all the occurrences of text that ends with .com
        remove_link = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
        return remove_link

    def remove_whitespace(self, text):
        pattern = re.compile(r'\s+') 
        Without_whitespace = re.sub(pattern, ' ', text)
        # There are some instances where there is no space after '?' & ')', 
        # So I am replacing these with one space so that It will not consider two words as one token.
        text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
        return text

    def removing_special_characters(self, text):
        # The formatted text after removing not necessary punctuations.
        Formatted_Text = re.sub(r"[^a-zA-Z0-9:$-,%.?!|]+", ' ', text) 
        # In the above regex expression,I am providing necessary set of punctuations that are frequent in this particular dataset.
        return Formatted_Text

    def accented_characters_removal(self, text):
        # Remove accented characters from text using unidecode.
        # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
        text = unidecode.unidecode(text)
        return text

    def replace_slangs_remove_stopwords(self, text):
        # Tokenizing text into tokens.
        list_Of_tokens = text.split(' ')

        stopwords_exist = []

        for i in range(len(list_Of_tokens)):
            item = list_Of_tokens[i]
        if item in self.slangs:
            # If Word is present in slangs, replace that word with the value.
            list_Of_tokens[i] = self.slangs.get(item)
            item = list_Of_tokens[i]

        if item in self.stopwords: 
            # If word is present in stopwords, add to list
            stopwords_exist.append(item)
                        
        list_Of_tokens = [word for word in list_Of_tokens if word not in stopwords_exist]

        # Converting list of tokens to String.
        String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens)
        return String_Of_tokens

    def stopwords_elimination(self, text, stopwords):
        text_list = [word for word in text if word not in stopwords]
        return ''.join(str(e) for e in text_list)

    def text_cleansing(self, tweet_post):
        tweet_post = tweet_post.str.lower()
        tweet_post = tweet_post.apply(self.remove_newlines_tabs)
        tweet_post = tweet_post.apply(self.strip_html_tags)
        tweet_post = tweet_post.apply(self.remove_links)
        tweet_post = tweet_post.apply(self.remove_whitespace)
        tweet_post = tweet_post.apply(self.removing_special_characters)
        tweet_post = tweet_post.apply(self.accented_characters_removal)
        lemma = Lemmatizer()
        tweet_post = tweet_post.apply(lemma.stem)
        tweet_post = tweet_post.apply(self.replace_slangs_remove_stopwords).dropna()
        return tweet_post