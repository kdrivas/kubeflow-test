import re
from spacy.lang.es.stop_words import STOP_WORDS
from nltk.corpus import stopwords
import re
import string
from textblob import TextBlob
# import spacy
from wordcloud import WordCloud
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.util import ngrams
import numpy as np
import unicodedata
from bs4 import BeautifulSoup
# import es_core_news_md
# nlp = es_core_news_md.load()
# nlp=spacy.load('es_core_news_md')
import pandas as pd
# nlp=pd.read_pickle('gs://ue4_ndlk_nonprod_stg_gcs_iadev_adsto/tmp/Ronald/obj_nlp.pkl')
# from nrclex import NRCLex
# import ast
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from textblob import TextBlob
# from afinn import Afinn
emoj = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002500-\U00002BEF"  # chinese char
    u"\U00002702-\U000027B0"
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u"\U00010000-\U0010ffff"
    u"\u2640-\u2642" 
    u"\u2600-\u2B55"
    u"\u200d"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\ufe0f"  # dingbats
    u"\u3030"
                  "]+", re.UNICODE)
list_stopwords=list(set(STOP_WORDS).union(stopwords.words('spanish')).union(stopwords.words('english')))
def clean_text(x):
    try:
        x = re.sub(emoj, ' ', x) # emoji
        x = re.sub('\[.*?¿\]\%', ' ', x) # word in brackets
        x = re.sub("\[.*?\]"," ",x) # word in brackets
        x = re.sub("\<.*?\>"," ",x) # word in btw
        x = re.sub("\(.*?\)"," ",x) # word in parenthesis
        x = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', ' ', x) #url
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore') # weird caracters
        x = BeautifulSoup(x, 'html.parser').get_text() #tags
        x = re.sub('\S*@\S*', ' ', x) #email
        x = re.sub('@\S*', ' ', x) #mention
        x = re.sub('#\S+', ' ', x) #hashtag
        x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x) #punctuation
        x = re.sub('\w*\d\w*', ' ', x) #numbers
        x = re.sub('[‘’“”…«»]', ' ', x) 
        x = re.sub(r'[^\w]', ' ', x)
        x = x.translate(str.maketrans('áéíóúüñÁÉÍÓÚÜ','aeiouunAEIOUU')) # remove tildes
        x = re.sub(r'^\s*|\s\s*', ' ', x).strip() # whitespaces
        x= ' '.join([t.strip().lower() for t in x.split(' ') if t.strip().lower() not in list_stopwords])
        # x= ' '.join([t.strip().lower() for t in x.split(' ')])
        return x
    except Exception as e: return e