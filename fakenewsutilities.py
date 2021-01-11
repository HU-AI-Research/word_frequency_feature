import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt

import pdb
import seaborn as sns
from nltk.corpus import stopwords

#use NLTK get the stopwords
stopwordslist = set(stopwords.words('english'))

#strip punctuation from string for future use
punct_regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~')) 

#make a function to clean the tweets text
def wash_pandas_str( input_df ):
    ret_text = input_df['text'].str.replace(r'…', '')
    ret_text = ret_text.str.replace(u'\u2019', '')
    ret_text = ret_text.str.replace(r'https\S*?\s', ' ')  
    ret_text = ret_text.str.replace(r'https\S*?$', '')
    ret_text = ret_text.str.replace(r'RT\s', '')
    ret_text = ret_text.str.replace(r'\s$', '')
    ret_text = ret_text.str.replace(r'@\S*?\s', '')
    ret_text = ret_text.str.replace(r'@\S*?$', '')
    
    ret_text = ret_text.str.replace('“', '')
    ret_text = ret_text.str.replace('--', '')
    ret_text = ret_text.str.replace('-', ' ')


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



#generate the high frequency words map
#if use ipython --pylab in Visual Studio Code can get the high quality image for word frequency
def plot_word_map(input_word_df_full):

    X = input_word_df_full['with_fake_prob']
    Y = input_word_df_full['cnt_in_true'] + input_word_df_full['cnt_in_fake']
    labels = input_word_df_full['word'].tolist()

    fig, ax = plt.subplots(figsize=(30, 30))
    ax.scatter(X, Y)

    for i in range(len(labels)):
        plt.text(X[i], Y[i], labels[i])

    fig.show()
 

#make a function, train the data by using Naive Bayes.    
#To save testing time, I've tried with limit in 500, 1000,1500,2000,2500 and 3000 high frequency words
#By choosing 2000 high frequency words, we can save testing time while maintaining a high level of accuracy. 
#If try higher value, we can get higher accuracy. I've tried with 10000 words, running time is 36mins, and get the accuracy at 0.96, very close to test with all words. 

def naive_bayes_train(X_train, Y_train, limit = 2000):
    
    #count the true tweets and high tweets numbers.
    fake_cnt = len(Y_train[Y_train == 1].index)
    true_cnt = len(Y_train[Y_train == 0].index)

    #get the priori probability of fake tweet.
    fake_prob_prior = fake_cnt / (fake_cnt + true_cnt)

    #{word：(cnt_in_true, cnt_in_fake),}, cnt_in_true/fake means the number of word occurrences in true/fake tweet
    ret_dict = {}

    for ind in X_train.index:
        twit_txt = punct_regex.sub('', X_train['text'][ind])

        #use set() convert tweets words to tuple, no repetition
        for i in set(twit_txt.lower().split()):

            if i not in stopwordslist:
                if i not in ret_dict.keys():    # new word found
                    
                    if Y_train[ind] == 0:       # new word found in true tweet
                                                # because I need the cnt for future calculations, 0 & 1 will cause problem, so I add 1 & 2.
                        ret_dict[i] = [2,1]      
                    else:                       # new word found in fake tweet
                        ret_dict[i] = [1,2]      
                                                
                else:                           # old word found
                    if Y_train[ind] == 0:       # old word found in true tweet 
                        ret_dict[i][0] += 1      
                    else:                       # old word found in fake tweet
                        ret_dict[i][1] += 1      


    #[word, cnt_in_true, cnt_in_fake,freq_true, freq_fake,total_cnt]    
    train_df = pd.DataFrame.from_dict(ret_dict, orient = 'index')
    train_df = train_df.reset_index()
    train_df.columns = ['word', 'cnt_in_true', 'cnt_in_fake']

    train_df['freq_true'] = train_df['cnt_in_true'] / true_cnt
    train_df['freq_fake'] = train_df['cnt_in_fake'] / fake_cnt
    train_df['total_cnt'] = train_df['cnt_in_true'] + train_df['cnt_in_fake']

    #sort by the word occurrences number, get 500 words.
    train_df = train_df.sort_values(by = ['total_cnt'],ascending=False).iloc[0:limit,:]

    return train_df, fake_prob_prior


#Use the train df from above function, generate the "words feature"
def naive_bayes_generate_feature(train_df, fake_prob_prior,X_test,Y_test):
 
    #the most frequently occurring words
    words_set = set(train_df['word'])
    
    accurate_count = 0

    #the "words feature"--the probability of fake tweet, will be save to this list
    ret_list=[]

    j = 0

    for ind in X_test.index:

        twit_txt = punct_regex.sub('', X_test['text'][ind])
        fake_prob = fake_prob_prior        #priori probability of fake tweet
        true_prob = 1 - fake_prob_prior    #priori probability of true tweet

        for i in set(twit_txt.lower().split()):
            if i in words_set:             

                #train_df['word','cnt_in_true', 'cnt_in_fake','freq_true', 'freq_fake','total_cnt']
                #Probability of being a true tweet, and a fake tweet
                true_prob_temp = true_prob * train_df[train_df['word'] == i].iloc[0,3]
                fake_prob_temp = fake_prob * train_df[train_df['word'] == i].iloc[0,4]

                #Since the probability values become smaller when multiplied together, I changed the format
                true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
                fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

        ret_list.append(fake_prob)

        #if the probability of being a fake tweet larger than true tweet, predict it to be fake.
        pred = int(fake_prob > true_prob)

        #if the prediction is correct, count to accurate
        accurate_count += (Y_test[ind] == pred)  

        j += 1
        #as this function takes quite a few mins to completion, I add this print to show the process
        if j % 1000 == 0:
            print ('{0} tested, accuracy {1:3f}'.format( j, accurate_count/j) )

    return ret_list
  



#Bigram frequency
#To save testing time, I've tried with limit in 500, 1000,1500,2000,2500 and 3000 two_words
#By choosing 2000 high frequency two_words, we can save testing time while maintaining a high level of accuracy. 
#If try higher value, we can get higher accuracy.  
def naive_bayes_bigrm_train(X_train, Y_train, limit = 2000):
    
    #count the true tweets and high tweets numbers.
    fake_cnt = len(Y_train[Y_train == 1].index)
    true_cnt = len(Y_train[Y_train == 0].index)

    #get the priori probability of fake tweet.
    fake_prob_prior = fake_cnt / (fake_cnt + true_cnt)

    #{two_word：(cnt_in_true, cnt_in_fake),}, cnt_in_true/fake means the number of two_word occurrences in true/fake tweet
    ret_dict = {}

    for ind in X_train.index:
        tweet = punct_regex.sub('', X_train['text'][ind])

        tweet = tweet.lower()  
        tokens = nltk.word_tokenize(tweet)
        bigrm = list(nltk.bigrams(tokens))

        #use set() convert tweets words to tuple in two_words, no repetition

        for i in bigrm:
            if i not in ret_dict.keys():    # new two_words found
                if Y_train[ind] == 0:       # new two_words found in true tweet
                                            
                    ret_dict[i] = [2,1]      
                else:                       # new two_words found in fake tweet
                    ret_dict[i] = [1,2]      
                                                
            else:                           # old two_words found
                if Y_train[ind] == 0:       # old two_words found in true tweet 
                    ret_dict[i][0] += 1      
                else:                       # old two_words found in fake tweet
                    ret_dict[i][1] += 1      


    #[two_words, cnt_in_true, cnt_in_fake,freq_true, freq_fake,total_cnt]    
    train_df = pd.DataFrame.from_dict(ret_dict, orient = 'index')
    train_df = train_df.reset_index()
    train_df.columns = ['two_words', 'cnt_in_true', 'cnt_in_fake']

    train_df['freq_true'] = train_df['cnt_in_true'] / true_cnt
    train_df['freq_fake'] = train_df['cnt_in_fake'] / fake_cnt
    train_df['total_cnt'] = train_df['cnt_in_true'] + train_df['cnt_in_fake']

    #sort by the word occurrences number, get 500 words.
    train_df = train_df.sort_values(by = ['total_cnt'],ascending=False).iloc[0:limit,:]

    return train_df, fake_prob_prior


#Use the train df from above function, generate the "bigram feature"
def naive_bayes_generate_feature(train_df, fake_prob_prior,X_test,Y_test):
 
    #the most frequently occurring two_words
    words_set = set(train_df['two_words'])
    
    accurate_count = 0

    #the "bigram feature"--the probability of fake tweet, will be save to this list
    ret_list=[]

    j = 0

    for ind in X_test.index:

        tweet = punct_regex.sub('', X_test['text'][ind])
        tweet = tweet.lower()  
        tokens = nltk.word_tokenize(tweet)
        bigrm = list(nltk.bigrams(tokens))

        fake_prob = fake_prob_prior        #priori probability of fake tweet
        true_prob = 1 - fake_prob_prior    #priori probability of true tweet
   

        for i in bigrm:
            if i in words_set:             

                #train_df['word','cnt_in_true', 'cnt_in_fake','freq_true', 'freq_fake','total_cnt']
                #Probability of being a true tweet, and a fake tweet
                true_prob_temp = true_prob * train_df[train_df['two_words'] == i].iloc[0,3]
                fake_prob_temp = fake_prob * train_df[train_df['two_words'] == i].iloc[0,4]

                #Since the probability values become smaller when multiplied together, I changed the format
                true_prob = true_prob_temp / (fake_prob_temp + true_prob_temp)
                fake_prob = fake_prob_temp / (fake_prob_temp + true_prob_temp)

        ret_list.append(fake_prob)

        #if the probability of being a fake tweet larger than true tweet, predict it to be fake.
        pred = int(fake_prob > true_prob)

        #if the prediction is correct, count to accurate
        accurate_count += (Y_test[ind] == pred)  

        j += 1
        #as this function takes quite a few mins to completion, I add this print to show the process
        if j % 1000 == 0:
            print ('{0} tested, accuracy {1:3f}'.format( j, accurate_count/j) )

    return ret_list
  


