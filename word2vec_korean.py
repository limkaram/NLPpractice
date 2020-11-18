## 필요 모듈 임포트
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

## 네이버 영화 리뷰 데이터 다운 및 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")
train_data = pd.read_table('ratings.txt')

## document columns 데이터 비어있는 행 제거
train_data = train_data[train_data['document'].isnull() == False]

## 불용어 정의
stopword = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

## okt 형태소 분석기 활용 토큰화
okt = Okt()

tokenized_data = []
for sentences in train_data['document']:
    temp = okt.morphs(sentences, stem=True)
    temp = [word for word in temp if word not in stopword]
    tokenized_data.append(temp)

## 임베딩 벡터 구현
from gensim.models import Word2Vec

model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=5, workers=4, sg=0)
# size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# window = 컨텍스트 윈도우 크기
# min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# workers = 학습을 위한 프로세스 수
# sg = 0은 CBOW, 1은 Skip-gram.

## 결과 확인
print(model.wv.vectors.shape)
print(model.wv.most_similar('최민식'))