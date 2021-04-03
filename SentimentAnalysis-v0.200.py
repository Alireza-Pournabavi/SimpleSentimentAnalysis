# Sentiment Analysis(Project in Testing) - Version:v0.200

##############################################################################   
##############################################################################
from nltk.tokenize import WordPunctTokenizer, TweetTokenizer
from nltk.corpus import stopwords, wordnet, twitter_samples
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.tag import pos_tag
from nltk import classify

from tkinter import Tk, Frame, Text, Scrollbar, ttk, LabelFrame, Label
from tkinter import Button
from tkinter.filedialog import askopenfilename

import threading, time
import re
import random
import string
import pandas as pd

global sk_TwitterSample, sk_FacebookPost, sk_TwitterAirline, bol_TwitterSample, bol_FacebookPost, bol_TwitterAirline
bol_TwitterSample, bol_FacebookPost, bol_TwitterAirline = False, False, False


##############################################################################
##############################################################################

class RemovingRepeatingCharacter(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r"(\w*)(\w)\2(\w*)")
        self.repl = r'\1\2\3'

    def replace(self, word):
        if wordnet.synsets(word):
            # print("word:\t", word)
            return word
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            # print("repl_word:\t", repl_word)
            return repl_word


def filter_stop_words():
    return stopwords.words("english")


RepeatChar = RemovingRepeatingCharacter()
stop_words = filter_stop_words()
spell = SpellChecker()


##############################################################################
##############################################################################

class Normalizing:
    def __init__(self, get1):
        # input
        self.get1 = get1

    # Main run
    def main_normalizing(self):
        input_ = self.get1.lower()
        # print("before input_:\n", input_)

        # unwanted character
        unwanted_digit = ['1', '2', '3', '4', '5', '5', '6', '7', '8', '9', '0']
        for digit in unwanted_digit:
            input_ = input_.replace(digit, '')

        unwanted_punk = ['"', "'", '=', '@', '&', '%', '.', ',', ':', '\\', '$', '^', '<', '>', '!', '?', '{', '}', ';',
                         '\n', '\t', '(', ')', '[', ']', '/', '*', '+', '#', '\u200c', '\ufeff', '-', '_', '|']
        for punk in unwanted_punk:
            input_ = input_.replace(punk, '')

        # print("after input_:\n", input_)

        # tokenize
        tokenizer = WordPunctTokenizer()
        word_token = tokenizer.tokenize(input_)
        # print("word_token:\n", word_token)

        # stop word

        for word in stop_words:
            for i in range(len(word_token)):
                if word_token[i] == word:
                    word_token[i] = ''
        # print("word_token after filter with stop words:\n", word_token)

        # removing repeating character

        for i in range(len(word_token)):
            word_token[i] = RepeatChar.replace(word_token[i])
        # print("word_token after removing repeating character:\n", word_token)

        # spellchecker

        for i in range(len(word_token)):
            word_token[i] = spell.correction(word_token[i])
        print("word_token after spell checker:\n", word_token)

        return word_token


##############################################################################
##############################################################################

class NormalizingDatasets:

    # removing noise
    @staticmethod
    def removing_noise(tweet_token, stop_words):
        cleaned = []

        for w_token, tag in pos_tag(tweet_token):
            w_token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                             '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', w_token)
            w_token = re.sub('(@[A-Za-z0-9_]+)', "", w_token)
            if tag.startswith("NN"):
                pos = 'n'
            elif tag.startswith("VB"):
                pos = 'v'
            else:
                pos = 'a'

            lemmatizer = WordNetLemmatizer()
            w_token = lemmatizer.lemmatize(w_token, pos)

            if len(w_token) > 0 and w_token not in string.punctuation and w_token.lower() not in stop_words:
                cleaned.append(w_token.lower())

        return cleaned

    # Create Dataset
    @staticmethod
    def create_model(cleaned):
        for tweet in cleaned:
            yield dict([w_token, True] for w_token in tweet)

    # Monitor Dataset for "facebook post" and "twitter Airline"
    @staticmethod
    def Monitor(dataset):
        tokenizer = TweetTokenizer()
        cleaned = []

        for sentence in dataset:
            word_token = tokenizer.tokenize(sentence)
            for token in word_token:
                token.lower()
                token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                               '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
                token = re.sub('(@[A-Za-z0-9_]+)', "", token)
                if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                    token = token.lower()
                token = RepeatChar.replace(token)
                token = spell.correction(token)
                if token is not None:
                    cleaned.append(token)

        return cleaned

    # Main run for "Twitter Sample"
    @staticmethod
    def main_TwitterSample():
        # Time too Long
        positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
        # print("positive_tweet_tokens:\n", positive_tweet_tokens)
        negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
        # print("negative_tweet_tokens:\n", negative_tweet_tokens)

        positive_cleaned = []
        negative_cleaned = []

        # Time very very Long
        stop_words = filter_stop_words()
        for token in positive_tweet_tokens:
            positive_cleaned.append(NormalizingDatasets.removing_noise(token, stop_words))
        for token in negative_tweet_tokens:
            negative_cleaned.append(NormalizingDatasets.removing_noise(token, stop_words))

        # print("positive_cleaned:\n", positive_cleaned)
        # print("negative_cleaned:\n", negative_cleaned)

        positive_cleaned_model = NormalizingDatasets.create_model(positive_tweet_tokens)
        negative_cleaned_model = NormalizingDatasets.create_model(negative_tweet_tokens)

        # print("positive_cleaned_model:\n", positive_cleaned_model)
        # print("negative_cleaned_model:\n", negative_cleaned_model)

        positive_dataset = [(tweet_tokens, "Positive") for tweet_tokens in positive_cleaned_model]
        negative_dataset = [(tweet_tokens, "Negative") for tweet_tokens in negative_cleaned_model]

        # print("positive_dataset:\n", positive_dataset)
        # print("negative_dataset:\n", negative_dataset)

        dataset = positive_dataset + negative_dataset

        # print("dataset:\n", dataset[:9])

        random.shuffle(dataset)

        train_dataset = dataset[:7000]
        test_dataset = dataset[7000:]

        # Classifier
        global sk_TwitterSample
        sk_TwitterSample = SklearnClassifier(MultinomialNB())
        sk_TwitterSample.train(train_dataset)

        # Accuracy
        print("Twitter Sample size is 10000")
        print("Accuracy Twitter Sample corpus is:", classify.accuracy(sk_TwitterSample, test_dataset))

    # Main run for "Facebook Post"
    @staticmethod
    def main_FacebookPost():
        fb = pd.read_csv('C:\\Users\\Alire\\Desktop\\NLP-Project\\facebook_dataset.csv')
        # print(fb)

        normalize = NormalizingDatasets()

        # Positive dataset
        pos = fb[['FBPost', 'Label']]
        print("Facebook Post dataset shape:", pos.shape)
        pos = pos[fb['Label'].str.find('P') != -1]
        pos_dataframe = pos['FBPost']
        # print(pos.shape)
        pos_cleaned = normalize.Monitor(pos_dataframe[:20])
        # print("pos_cleaned:\n", pos_cleaned)
        pos_model = normalize.create_model(pos_cleaned)
        # print("pos_model:\n", pos_model)
        pos_dataset = [(tokens, "Positive") for tokens in pos_model]
        # print("pos_dataset:\n", pos_dataset)

        # Negative dataset
        neg = fb[['FBPost', 'Label']]
        neg = neg[fb['Label'].str.find('N') != -1]
        neg_dataframe = neg['FBPost']
        # print(neg.shape)
        neg_cleaned = normalize.Monitor(neg_dataframe)
        neg_model = normalize.create_model(neg_cleaned)
        neg_dataset = [(tokens, "Positive") for tokens in neg_model]

        # Neutral dataset
        neu = fb[['FBPost', 'Label']]
        neu = neu[fb['Label'].str.find('O') != -1]
        neu_dataframe = neu['FBPost']
        # print(neu.shape)
        neu_cleaned = normalize.Monitor(neu_dataframe)
        neu_model = normalize.create_model(neu_cleaned)
        neu_dataset = [(tokens, "Positive") for tokens in neu_model]

        dataset_facebook = pos_dataset + neg_dataset + neu_dataset
        # print("pos dataset:\n", dataset_facebook)

        random.shuffle(dataset_facebook)

        train_facebook_dataset = dataset_facebook[:700]
        test_facebook_dataset = dataset_facebook[700:]

        # Classifier
        global sk_FacebookPost
        sk_FacebookPost = SklearnClassifier(MultinomialNB())
        sk_FacebookPost.train(train_facebook_dataset)

        # Accuracy
        print("Accuracy Facebook Post dataset is:", classify.accuracy(sk_FacebookPost, test_facebook_dataset))

    # Main run for "Twitter Airline"
    @staticmethod
    def main_TwitterAirline():
        tweet_airline = pd.read_csv('C:\\Users\\Alire\\Desktop\\NLP-Project\\Tweets.csv')
        print("Twitter Airline dataset shape:", tweet_airline.shape)
        # print(tweet_airline)

        normalize = NormalizingDatasets()

        # Positive dataset
        pos = tweet_airline[['text', 'airline_sentiment']]
        pos = pos[tweet_airline['airline_sentiment'].str.find('positive') != -1]
        pos_dataframe = pos['text']
        # print(pos.shape)
        pos_cleaned = normalize.Monitor(pos_dataframe[:20])
        # print("pos_cleaned:\n", pos_cleaned)
        pos_model = normalize.create_model(pos_cleaned)
        # print("pos_model:\n", pos_model)
        pos_dataset = [(tokens, "Positive") for tokens in pos_model]
        # print("pos_dataset:\n", pos_dataset)

        # Negative dataset
        neg = tweet_airline[['text', 'airline_sentiment']]
        neg = neg[tweet_airline['airline_sentiment'].str.find('negative') != -1]
        neg_dataframe = neg['text']
        # print(neg.shape)
        neg_cleaned = normalize.Monitor(neg_dataframe)
        neg_model = normalize.create_model(neg_cleaned)
        neg_dataset = [(tokens, "Positive") for tokens in neg_model]

        # Neutral dataset
        neu = tweet_airline[['text', 'airline_sentiment']]
        neu = neu[tweet_airline['airline_sentiment'].str.find('neutral') != -1]
        neu_dataframe = neu['text']
        # print(neu.shape)
        neu_cleaned = normalize.Monitor(neu_dataframe)
        neu_model = normalize.create_model(neu_cleaned)
        neu_dataset = [(tokens, "Positive") for tokens in neu_model]

        dataset_TweetAirline = pos_dataset + neg_dataset + neu_dataset
        # print("pos dataset:\n", dataset_TweetAirline[:20])

        random.shuffle(dataset_TweetAirline)

        train_TwitterAirline_dataset = dataset_TweetAirline[:12000]
        test_TwitterAirline_dataset = dataset_TweetAirline[12000:]

        # Classifier
        global sk_TwitterAirline
        sk_TwitterAirline = SklearnClassifier(MultinomialNB)
        sk_TwitterAirline.train(train_TwitterAirline_dataset)

        # Accuracy
        print("Accuracy Twitter Airline dataset is:", classify.accuracy(sk_TwitterAirline, test_TwitterAirline_dataset))

    # final output method
    @staticmethod
    def output_NormalizingDatasets(word_token):
        # Final Answer
        b = sk_FacebookPost.classify(dict([token, True] for token in word_token))
        return b


'''
sample input 1: "I ordered just once from TerribleCo, they screwed up, never used the app again."   
--> Negative
sample input 2: "Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. 
#thanksGenericAirline"   
--> Positive
'''


##############################################################################
##############################################################################


class GUITkinter:
    def __init__(self, win):
        win = win
        win.title("Sentiment Analysis")
        win.iconbitmap(
            'C:/Users/Alire/Downloads/Compressed/airpournabavi_oaz_1.ico')
        win.resizable(False, False)

        frame_text = Frame(win)
        frame_progressbar = Frame(win)
        frame_process = Frame(win, pady=18)
        frame_Buttons = Frame(win, padx=5)
        frame_answer = Frame(win)
        frame_note = LabelFrame(win,
                                text='Note', labelanchor='n', pady=5)

        frame_text.pack(fill='both', expand=1)
        frame_progressbar.pack(fill='both', expand=1)
        frame_process.pack(fill='both', expand=1)
        frame_Buttons.pack(fill='both', expand=1)
        frame_answer.pack(fill='both', expand=1)
        frame_note.pack(fill='x', expand=1)

        # Text
        self.text = Text(frame_text)
        self.text.pack(side='left', fill='x', expand=1)
        sc = Scrollbar(frame_text, command=self.text.yview)
        sc.pack(side='right', fill='y')
        self.text.config(yscrollcommand=sc.set)

        # Progress Bar
        self.pb = ttk.Progressbar(frame_progressbar,
                                  mode='indeterminate', length=1090)
        self.pb.pack(expand=1)
        self.lb_progressbar = Label(frame_progressbar,
                                    text='Not yet Processing!!!', pady=18)
        self.lb_progressbar.pack()

        # Process
        self.cb_process = ttk.Combobox(frame_process, values=(
            'Twitter Sample', 'Facebook Post', 'Twitter Airline'), state='readonly')
        self.cb_process.pack(side='left', expand=1)
        self.btn_cb = Button(frame_process,
                             text='Start Dataset Process')
        self.btn_cb.pack(side='right', expand=1)
        self.btn_cb.bind('<Button>', self.select_dataset)

        # Buttons
        self.btn_run = Button(frame_Buttons, text='Run', padx=105, pady=18)
        self.btn_run.pack(side='left', expand=1)
        self.btn_run.bind('<Button>', self.input_run)
        self.btn_log = Button(frame_Buttons, text='Log File',
                              padx=105, pady=18, state='disable')
        self.btn_log.pack(side='left', expand=1)
        self.btn_openfile = Button(
            frame_Buttons, text='Open File',
            padx=105, pady=18)
        self.btn_openfile.pack(side='left', expand=1)
        self.btn_openfile.bind('<Button>',
                               self.input_openfile)

        # Answer
        self.lb_buttons = Label(
            frame_answer, text="None", pady=18, font='bold')
        self.lb_buttons.pack(side='bottom')

        # Note
        tx = """سلام خدمت شما کاربر عزیز. با تشکر از اینکه این برنامه را استفاده میکنید مقداری توضیحات و بیان نکته 
        هایی را به خدمتتان میرسانم. این برنامه تحلیلگر احساسات متن یا همان Sentiment Analysis میباشد که به دو صورت 
        متنی که در باکس قرار داده شده و یا گرفتن فایل تکست میتوان به آن ورودی داد و پس از آن ماشین شروع به کار میکند 
        و نتیجه در پنجره قابل مشاهده خواهد بود. این برنامه نسخه اولیه و آزمایشی آن را سپری میکند پس واضح است استثنا 
        ها برای آن پیش بینی نشده است اذا خواهشمندم علاوه بر تست کردن ماشین و ارسال گزارش های آن به من برنامه را هم 
        تست کنید. با تشکر علیرضا پورنبوی Sentiment Analysis(Project in Testing) - Version:v0.1 Email: 
        ap1375.11@gmail.com """
        lb_note = Label(frame_note, text=tx, font=('times', 10))
        lb_note.pack()

    # send Text
    def input_send(self, txt):
        # print('txt:\n', txt)
        NO = Normalizing(txt)
        ND = NormalizingDatasets()
        mn = NO.main_normalizing()
        ot = ND.output_NormalizingDatasets(mn)
        self.input_receive(ot)

    # input with open file
    def input_openfile(self, event):
        address_file = askopenfilename(filetypes=(
            ('Text File', '*.txt'), ('All Files', '*.txt')))
        file = open(address_file, encoding='utf-8')
        txt = file.read()
        self.pb.start()
        self.lb_progressbar.config(text="Processing...!!!")
        self.lb_buttons.config(text='None')
        self.btn_run.config(state='disable')
        # self.btn_log.config(state='disable')
        self.btn_openfile.config(state='disable')
        self.btn_cb.config(state='disable')
        th = threading.Thread(target=self.input_send,
                              args=(txt,))
        th.start()

    # input with text box
    def input_run(self, event):
        txt = self.text.get('1.0', 'end')
        self.pb.start()
        self.lb_progressbar.config(text="Processing...!!!")
        self.lb_buttons.config(text='None')
        self.btn_run.config(state='disable')
        # self.btn_log.config(state='disable')
        self.btn_openfile.config(state='disable')
        self.btn_cb.config(state='disable')
        th = threading.Thread(target=self.input_send,
                              args=(txt,))
        th.start()

    # Give Answer
    def input_receive(self, answer):
        if answer == 'Negative':
            self.lb_buttons.config(text=answer)
        elif answer == 'Positive':
            self.lb_buttons.config(text=answer)
        self.pb.stop()
        self.lb_progressbar.config(text='Processed')
        self.btn_run.config(state='active')
        # self.btn_log.config(state='active')
        self.btn_openfile.config(state='active')
        self.btn_cb.config(state='active')

    # Select Dataset
    def select_dataset(self, event):
        global sk_TwitterSample, sk_FacebookPost, sk_TwitterAirline, bol_TwitterSample, bol_FacebookPost, bol_TwitterAirline
        dataset = self.cb_process.get()
        if dataset == 'Twitter Sample' and bol_TwitterSample == False:
            bol_TwitterSample = True
            self.pb.start()
            self.lb_progressbar.config(
                text="Corpus is Loading. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_send, args=(1,))
            th.start()
        elif dataset == 'Facebook Post' and bol_FacebookPost == False:
            bol_FacebookPost = True
            self.pb.start()
            self.lb_progressbar.config(
                text="Dataset is Loading. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_send, args=(2,))
            th.start()
        elif dataset == 'Twitter Airline' and bol_TwitterAirline == False:
            bol_TwitterAirline = True
            self.pb.start()
            self.lb_progressbar.config(
                text="Dataset is Loading. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_send, args=(3,))
            th.start()
        ###
        elif dataset == 'Twitter Sample' and bol_TwitterSample == True:
            self.lb_progressbar.config(
                text="This Corpus has already been processed. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_recieve, args=(False,))
            th.start()
        elif dataset == 'Facebook Post' and bol_FacebookPost == True:
            self.lb_progressbar.config(
                text="This Dataset has already been processed. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_recieve, args=(False,))
            th.start()
        elif dataset == 'Twitter Airline' and bol_TwitterAirline == True:
            self.lb_progressbar.config(
                text="This Dataset has already been processed. Please Wait!!!")
            self.btn_run.config(state='disable')
            # self.btn_log.config(state='disable')
            self.btn_openfile.config(state='disable')
            self.btn_cb.config(state='disable')
            th = threading.Thread(target=self.dataset_recieve, args=(False,))
            th.start()

    # Dataset Send
    def dataset_send(self, n):
        if n == 1:
            NormalizingDatasets.main_TwitterSample()
        elif n == 2:
            NormalizingDatasets.main_FacebookPost()
        elif n == 3:
            NormalizingDatasets.main_TwitterAirline()
        self.dataset_recieve()

    # Dataset Recieve
    def dataset_recieve(self, bol =True):
        if bol == False:
            time.sleep(2)
        self.pb.stop()
        self.lb_progressbar.config(text="Ready!")
        self.btn_run.config(state='active')
        # self.btn_log.config(state='active')
        self.btn_openfile.config(state='active')
        self.btn_cb.config(state='active')


##############################################################################
##############################################################################

if __name__ == "__main__":
    root = Tk()
    Main_GUI = GUITkinter(root)
    root.mainloop()

##############################################################################
##############################################################################
