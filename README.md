# word_frequency_feature

2020.12.28 :
After I finished the week6 assignment, I noticed the functions I made ih fakenews_utilities.py can be done by sklearn.feature_extraction.text.CountVectorizer directly.
You can ignore the last word_frequency notebook, check the new one. ;)

Using Naïve Bayes, the model accuracy is 96.6%.



---------------------------------------------------------------------------------------------------

2020.12.21 : 
In fakenews_utilities.py, there are 4 functiones, which I imported to word_frequency notebook.

1. def wash_pandas_str is to clean the tweets text.

2. def generate_word_df_from_label(input_frame, label = ?) is to make " count of tweets containing a typical word",

3. def generate_word_df_full(twit_df, fake_words_df, true_words_df, limit = 500) is to compare word frequency count in true tweets and fake tweets , 
and the probability of a tweet be a fake tweet when it contains or not contains a word. 
(limit means the most 500 high frequency words, can also change to 800 or more.)

4. def plot_word_map()  is to make a word frequency map of words in true and fake tweets, just to provide an intuitive word frequency visualization.


The result from function3 will be developed into a feature in next step. 
Aim to predict a tweet to be true or fake by rate all words in the tweet base on "probabiliy of that word in fake tweet".




 
