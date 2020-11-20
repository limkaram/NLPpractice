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
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
# urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
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
test_data = test_data.dropna(how='any')

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

## 훈련 데이터 등장 빈도수 분포 확인
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)

##
threshold = 3
total_cnt = len(tokenizer.word_index)  # 단어의 수
rare_cnt = 0  # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0  # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0  # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if value < threshold:
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s' %(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100)

##
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :', vocab_size)

##
tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

##
y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

## 빈 샘플들을 제거
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)

## 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

## 최적의 max_len 도출
def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len = 30
below_threshold_len(max_len, X_train)

## 패딩
<<<<<<< HEAD
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
=======
X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

## LSTM 활용 감성 분류
from tensorflow.keras.layers import Embedding, Dense, LSTM, Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

##
embedding_dim = 128
dropout_prob = (0.5, 0.8)
num_filters = 128

##
model_input = Input(shape = (max_len,))
z = Embedding(vocab_size, embedding_dim, input_length = max_len, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)

conv_blocks = []

for sz in [3, 4, 5]:
    conv = Conv1D(filters = num_filters,
                         kernel_size = sz,
                         padding = "valid",
                         activation = "relu",
                         strides = 1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)

z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(128, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('CNN_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), verbose=2, callbacks=[es, mc])
#
# filters = [3, 4, 5]
# conv_models = []
# for filter in filters:
#   conv_feat = Conv1D(filters=100,
#                             kernel_size=filter,
#                             activation='relu',
#                             padding='valid')(seq_embedded) # Convolution Layer
#   pooled_feat = GlobalMaxPooling1D()(conv_feat)  # MaxPooling
#   flatten_feat = Flatten()(pooled_feat)
#   conv_models.append(flatten_feat)
#
# conv_merged = Concatenate(conv_models)  # filter size가 2,3,4,5인 결과들 Concatenation
#
# model_output = Dropout(0.5)(conv_merged)
# model_output = Dense(10, activation='relu')(model_output)
# logits = Dense(1, activation='sigmoid')(model_output)
#
# model = Model(seq_input, logits)  # (입력,출력)
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.summary()
#
# #학습 시작
# history = model.fit(X_train, y_train,
#                     epochs=10,
#                     verbose=True,
#                     validation_data=(X_test, y_test),
#                     batch_size=128)
>>>>>>> 6ad97e9a12412378879bf8cdbefd4f0b759b70a7
