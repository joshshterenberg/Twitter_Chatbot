import tweepy
import time
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)
with open('intents.json') as file:
    data = json.load(file)

consumer_key = '8hRxVfqSOYcWDGVv2jqK6ewlX' #API Key
consumer_secret = 'rO8aKnP5HMXYsDT3Ua49jAhNchSSSiA8JymG4YYyeBht8z1gWB' #API Secret Key
key = '1291796120688001024-mycXVI2d6XmlXpJc7woZkpPAsopBIG' # Access Token Key
secret = 'mMz2cVy31JQbuWyC5LFgnmkGAgmR0Gdd46Tt5lwhLp3Zw' # Access Token Secret Key

# Authorizes and sets everything up
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(key, secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Helpers for reading and writing to last_seen file (so we don't concern with old tweets)
FILE_NAME = 'last_seen.txt'
def read_last_seen(FILE_NAME):
    file_read = open(FILE_NAME, 'r')
    last_seen_id = int(file_read.read().strip())
    file_read.close()
    return last_seen_id
def store_last_seen(FILE_NAME, last_seen_id):
    file_write = open(FILE_NAME, 'w')
    file_write.write(str(last_seen_id))
    file_write.close()
    return

# BUILDING TF MODEL
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    sords = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

# TRAINING / SAVING TF MODEL
tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1250, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# DEPLOYMENT
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def reply():
    # Gets all tweets that mention the Bot in reverse chronological order, if we haven't seen them before
    tweets = api.mentions_timeline(read_last_seen(FILE_NAME), tweet_mode="extended")

    for tweet in reversed(tweets):
        # if '#ultimatebot' in tweet.full_text.lower(): # if the tweet contains the hashtag
        #     print(str(tweet.id) + ' - ' + tweet.full_text) # print it
        #     api.update_status("@" + tweet.user.screen_name + " Auto reply, like and retweet work :)", tweet.id)
        #     api.create_favorite(tweet.id)
        #     api.retweet(tweet.id)
        #     store_last_seen(FILE_NAME, tweet.id)
        user = api.get_user(tweet.user.screen_name)
        friends = api.followers_ids()
        if '#josh' in tweet.full_text.lower(): #if the tweet contains Josh (dm contact details)
            if user.id in friends:
                api.send_direct_message(user.id, "You can contact my creator (Josh) by:\nEmail: joshuashterenberg@gmail.com\nInstagram: josh.strmntn")
            else:
                pass
        api.create_favorite(tweet.id)
        raw_text = tweet.full_text
        while "#" in raw_text:
            hashIndex = raw_text.find("#")
            while True:
                if (raw_text[hashIndex] == " ") or (hashIndex == len(raw_text)-1):
                    break
                else:
                    raw_text = raw_text[:hashIndex] + raw_text[hashIndex+1:]
            raw_text = raw_text[:hashIndex] + raw_text[hashIndex+1:]
        while "@" in raw_text:
            hashIndex = raw_text.find("@")
            while True:
                if (raw_text[hashIndex] == " ") or (hashIndex == len(raw_text)-1):
                    break
                else:
                    raw_text = raw_text[:hashIndex] + raw_text[hashIndex+1:]
            raw_text = raw_text[:hashIndex] + raw_text[hashIndex+1:]
        # raw_text is now the optimized text for machine learning model
        inp = raw_text
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if '#josh' not in tweet.full_text.lower():
            if results[results_index] > 0.65:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                api.update_status("@" + tweet.user.screen_name + " " + random.choice(responses), tweet.id)
            else:
                api.update_status("@" + tweet.user.screen_name + " I didn't understand that. Try again, or contact my creator by following me and replying #Josh to this tweet.", tweet.id)
        store_last_seen(FILE_NAME, tweet.id)

while True:
    reply()
    time.sleep(15)
