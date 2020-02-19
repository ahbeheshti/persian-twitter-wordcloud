import re
import codecs
import numpy as np
from twython import Twython
from PIL import Image
from hazm import *
import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud_fa import WordCloudFa

# Params

TWITTER_APP_KEY = "#########################"
TWITTER_APP_SECRET = "##################################################"

USER_NAME = "username"
TWEET_COUNT = 200

BACKGROUND_COLOR = "white"
COLOR_MAP = "autumn"
FONT_PATH = "XTitre.TTF"
MASK_PATH = "mask.png"

STOPWORDS_PATH = "stopwords.dat"
SAVE_PATH = "word_cloud.png"

# Get timeline
twitter = Twython(TWITTER_APP_KEY, TWITTER_APP_SECRET)
user_timeline = twitter.get_user_timeline(screen_name=USER_NAME, count=TWEET_COUNT)
raw_tweets = []
for tweets in user_timeline:
    raw_tweets.append(tweets['text'])
print(raw_tweets)
print(len(raw_tweets))

# Normalize words
tokenizer = WordTokenizer()
lemmatizer = Lemmatizer()
normalizer = Normalizer()
stopwords = set(list(map(lambda w: w.strip(), codecs.open(STOPWORDS_PATH, encoding='utf8'))))
words = []
for raw_tweet in raw_tweets:
    raw_tweet = re.sub(r"[,.;:?!،()]+", " ", raw_tweet)
    raw_tweet = re.sub('[^\u0600-\u06FF]+', " ", raw_tweet)
    raw_tweet = re.sub(r'[\u200c\s]*\s[\s\u200c]*', " ", raw_tweet)
    raw_tweet = re.sub(r'[\u200c]+', " ", raw_tweet)
    raw_tweet = re.sub(r'[\n]+', " ", raw_tweet)
    raw_tweet = re.sub(r'[\t]+', " ", raw_tweet)
    raw_tweet = normalizer.normalize(raw_tweet)
    raw_tweet = normalizer.character_refinement(raw_tweet)
    tweet_words = tokenizer.tokenize(raw_tweet)
    tweet_words = [lemmatizer.lemmatize(tweet_word).split('#', 1)[0] for tweet_word in tweet_words]
    tweet_words = list(filter(lambda x: x not in stopwords, tweet_words))
    words.extend(tweet_words)
print(words)

# Build word_cloud
mask = np.array(Image.open(MASK_PATH))
clean_string = ' '.join([str(elem) for elem in words])
clean_string = arabic_reshaper.reshape(clean_string)
clean_string = get_display(clean_string)
word_cloud = WordCloudFa(persian_normalize=False, mask=mask, colormap=COLOR_MAP,
                         background_color=BACKGROUND_COLOR, include_numbers=False, font_path=FONT_PATH)
wc = word_cloud.generate(clean_string)
image = wc.to_image()
image.show()
image.save(SAVE_PATH)
