import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt

import pdb

from nltk.corpus import stopwords

#use NLTK get the stopwords
stopwordslist = set(stopwords.words('english'))

#strip punctuation from string for future use
punct_regex = re.compile('[%s]' % re.escape(string.punctuation))

#make a function to clean the tweets text
 
def wash_pandas_str( input_df ):
    ret_text = input_df['text'].str.replace(r'…', '')
    ret_text = ret_text.str.replace('\'', '')

    ret_text = ret_text.str.replace(r'https\S*?\s', ' ')  
    ret_text = ret_text.str.replace(r'https\S*?$', '')
    ret_text = ret_text.str.replace(r'RT\s', '')
    ret_text = ret_text.str.replace(r'\s$', '')

    ret_text = ret_text.str.replace(r'@\S*?\s', '')
    ret_text = ret_text.str.replace(r'@\S*?$', '')

    input_df['text'] = ret_text
    return input_df

#make a function to generate a dataframe according to the label:
def generate_word_df_from_label(input_frame, label = 1):
    #make a dictionary first (convert it to dataframe later, which is more efficient)
    ret_dict = {}

    ret_df = input_frame[input_frame['label'] == label]

    #clean the tweet text, then loop through each word of the tweet txt 
    #if a word not include in stopwords, then add to ret_dic.  
    for ind in ret_df.index:
        twit_txt = punct_regex.sub('', ret_df['text'][ind])
        for i in set(twit_txt.lower().split()):

            if i not in stopwordslist:
                if i not in ret_dict.keys():
                    ret_dict[i] = 1
                else:
                    ret_dict[i] += 1

    #convert the dictionary to a dataframe, then reset its index
    words_df = pd.DataFrame.from_dict( ret_dict, orient = 'index')
    words_df = words_df.reset_index()

    #when label ==1, I'm checking only the fake news tweets.
    #words_df columns will be the "word","fake tweets count containing that word","fake tweets count include that word / total fake tweets count".
    if label == 1:
        words_df.columns = ['word', 'containing_twits_in_fake']
        words_df = words_df.sort_values(by=['containing_twits_in_fake'],ascending = False)
        words_df['freq'] = words_df['containing_twits_in_fake'] / len(ret_df.index)
    else:
        words_df.columns = ['word', 'containing_twits_in_true']
        words_df = words_df.sort_values(by=['containing_twits_in_true'],ascending = False)
        words_df['freq'] = words_df['containing_twits_in_true'] / len(ret_df.index)

    return words_df

    
#make a function,to get the most high frequency words in fake and true tweets    
def generate_word_df_full(twit_df, fake_words_df, true_words_df, limit = 500):
    fake_df = twit_df[twit_df['label'] == 1]
    true_df = twit_df[twit_df['label'] == 0]

    fake_twit_cnt = len(fake_df.index)
    true_twit_cnt = len(true_df.index)

    total_twit_cnt = fake_twit_cnt + true_twit_cnt

    #use set() convert words to tuple, no repetition
    true_words_set = set(true_words_df['word'])

    #ret_df = pd.DataFrame(columns=['word', 'cnt_in_true', 'cnt_in_fake', 'freq_in_true', 'freq_in_fake', 'freq_diff', 'with_fake_prob', 'without_fake_prob'])
    ret_dict = {}

    i = 0
    #loop through the words in fake_words_df
    for ind in fake_words_df.index:
        if i >= limit:
            break

        word = fake_words_df['word'][ind]
        cnt_in_fake = fake_words_df['containing_twits_in_fake'][ind]
        freq_in_fake = fake_words_df['freq'][ind]

        #then find the same word in true_words_df, find it directly in true_words_set, get the value of the word-containing tweets count and frequency
        #if can't find it in true_words_set, the return 0.
        if word in true_words_set:
            cnt_in_true = true_words_df[true_words_df['word'] == word].iloc[0,1]
            freq_in_true = true_words_df[true_words_df['word'] == word].iloc[0,2]
        else:
            cnt_in_true = 0
            freq_in_true = 0

        #this will be used to compare the frequency of words in fake and true tweets, if the frequencies are similar, there would be no reference value
        freq_diff = freq_in_fake - freq_in_true

        #the probability of a tweet is fake when containing a word 
        with_fake_prob = cnt_in_fake / (cnt_in_fake + cnt_in_true)
        #the probability of a tweet is fake when not containing a word
        without_fake_prob = (fake_twit_cnt - cnt_in_fake) / (total_twit_cnt - cnt_in_fake - cnt_in_true)

        #add above values to the dictionary
        ret_dict[word] = [cnt_in_true, cnt_in_fake, freq_in_true, freq_in_fake, freq_diff, with_fake_prob, without_fake_prob]

        i += 1
    
    #convert the dictionary to dataframe
    ret_df = pd.DataFrame.from_dict(ret_dict, orient = 'index')
    ret_df = ret_df.reset_index()
    ret_df.columns = ['word', 'cnt_in_true', 'cnt_in_fake', 'freq_in_true', 'freq_in_fake', 'freq_diff', 'with_fake_prob', 'without_fake_prob']
    return ret_df


#use ipython --pylab can get the high quality image for word frequency
def plot_word_map(input_word_df_full):

    X = input_word_df_full['with_fake_prob']
    Y = input_word_df_full['cnt_in_true'] + input_word_df_full['cnt_in_fake']
    labels = input_word_df_full['word'].tolist()

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.scatter(X, Y)

    for i in range(len(labels)):
        plt.text(X[i], Y[i], labels[i])

    fig.show()










