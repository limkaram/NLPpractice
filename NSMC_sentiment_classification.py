## module import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## 데이터 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

## 중복 데이터 제거
train_data.drop_duplicates(subset=['document'], inplace=True)
test_data.drop_duplicates(subset=['document'], inplace=True)

## NaN 데이터 제거
train_data = train_data.dropna(how='any')
test_data = test_data.dropna(how='any')

## label 분포 확인
import seaborn as sns

sns.countplot(x=train_data['label'])
sns.countplot(x=test_data['label'])
plt.show()

## 데이터 정제
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
# 한글과 공백을 제외하고 모두 제거

## 공백 데이터 제거
train_data['document'].replace('', np.nan, inplace=True)
test_data['document'].replace('', np.nan, inplace=True)

train_data = train_data.dropna(how='any')
test_data = train_data.dropna(how='any')

## 불용어 정의
# https://www.ranks.nl/stopwords/korean내 모든 불용어 제거
with open('stopwords_korean.txt', 'r', encoding='utf-8') as file:
    stopwords = file.readlines()
stopwords = [word.strip() for word in stopwords]

## 토큰화
okt = Okt()

X_train = []
for sentence in train_data['document']:
    temp = []
    temp = okt.morphs(sentence, stem=True)
    temp = [word for word in temp if word not in stopwords]
    X_train.append(temp)

##
X_test = []
for sentence in test_data['document']:
    temp = []
    temp = okt.morphs(sentence, stem=True)
    temp = [word for word in temp if word not in stopwords]
    X_test.append(temp)

##
print(len(X_train))
print(len(X_test))

##

