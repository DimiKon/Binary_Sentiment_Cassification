# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 13:41:26 2022

@author: dim_k
"""


# Import libraries

from glob import glob
import pandas as pd
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
#%%
# Raw dataset

input_path = './aclImdb'


# CSV outputs

train_path = './data_train.csv'
dev_path = './data_dev.csv'
test_path = './data_test.csv'

#%%

# Get list of text file paths

txt_paths_pos = glob(input_path + '/*/pos/*.txt') 
txt_paths_neg = glob(input_path + '/*/neg/*.txt')
print("Files found (positive reviews): ", len(txt_paths_pos))
print("Files found (negative reviews): ", len(txt_paths_neg))

#%%
# Helper functions
def get_text_from_file(path):
    with open(path, 'r', encoding="utf-8") as file:
        return file.read()

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


#%%
def preprocess_text(text):
    '''
    A function to clean datasets to be used for deep learning models
    '''
    
    # Removing html tags
    sentence = remove_tags(text)

    # Remove punctuations
    sentence = re.sub('[^a-zA-z0-9\s]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def clean_numbers(x):
    '''
    A function to replace numerals with # (because most embeddings have preprocessed their text like this)
    '''
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

def _get_contractions(contraction_dict):
    
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re

contractions, contractions_re = _get_contractions(contraction_dict)


def replace_contractions(text):
    '''
    A function to replace contractions with their respectve expanded forms
    '''
    def replace(match):
        return contractions[match.group(0)]
    return contractions_re.sub(replace, text)


#%%
# Create dataframe containing positive reviews
df_pos = pd.DataFrame({'review': txt_paths_pos})
df_pos.insert(loc=1, column='sentiment', value= 'positive')
# Create dataframe containing negative reviews
df_neg = pd.DataFrame({'review': txt_paths_neg})
df_neg.insert(loc=1, column='sentiment', value= 'negative')
# Concatenate dataframes containing positive and negative reviews
df = pd.concat([df_pos, df_neg])
# Replace path with review text
df['review'] = df['review'].apply(lambda path: get_text_from_file(path))
# Replace contractions by expanded forms
df['review'] = df['review'].apply(lambda text: replace_contractions(text))
# Clean text
df['review'] = df['review'].apply(lambda text: preprocess_text(text))
# Remove numerals
df['review'] = df['review'].apply(lambda text: clean_numbers(text))
# Binarize labels
df['sentiment']=df['sentiment'].replace(to_replace='negative',value=0)
df['sentiment']=df['sentiment'].replace(to_replace='positive',value=1)

#%%
# A function to split dataset in train, validation, test (shuffling and stratified sampling of datapoints)
def split_stratified_into_train_val_test(df_input, stratify_colname='sentiment',
                                         frac_train=0.82, frac_val=0.09, frac_test=0.09,
                                         random_state=1):
    
    
    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test

#%%


train, val, test = split_stratified_into_train_val_test(df)


print(train.sentiment.value_counts())

print(val.sentiment.value_counts())

print(test.sentiment.value_counts())

#%%
# Get some statistics regarding the lenght of reviews in dataset
df_stat = df.copy()
df_stat['word_count'] = df['review'].apply(lambda txt: len(txt.split(' ')))
#print(df_stat.head())

q=0.95
x = df_stat['word_count']
sns.distplot(x, hist=False, rug=True);
print('Minimum word count required to include all words in {}% of the reviews: {}'.format(q*100, x.quantile(q)))  
#%%
# Export dataframes to CSVs
train.to_csv(train_path, index = None, header=True)
val.to_csv(dev_path, index = None, header=True)
test.to_csv(test_path, index = None, header=True)


