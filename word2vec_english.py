## 필요 모듈 임포트
import re
from lxml import etree
import urllib.request
import zipfile
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('punkt')

## xml 데이터 로드 후 필요 데이터만 파싱
urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
# 데이터 다운로드

with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
  target_text = etree.parse(z.open('ted_en-20160408.xml', 'r'))
  parse_text = '\n'.join(target_text.xpath('//content/text()'))
# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.

## 텍스트 전처리 후 문장 토큰화 수행
content_text = re.sub(r'\([^)]*\)', '', parse_text)
sent_text = sent_tokenize(content_text)

## 구두점 제거
normalized_text = []
for string in sent_text:
  tokens = re.sub(r'[^a-z0-9]+', ' ', string.lower())
  normalized_text.append(tokens)

## 단어 토큰화 수행
result = [word_tokenize(word) for word in normalized_text]

## Word2Vec 학습
from gensim.models import Word2Vec

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)
# size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
# window = 컨텍스트 윈도우 크기
# min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
# workers = 학습을 위한 프로세스 수
# sg = 0은 CBOW, 1은 Skip-gram.

## word2vec 학습 결과 확인
model_result = model.wv.most_similar('guy')  # 특정 단어와 유사도가 높은 단어 반환
print(model_result)

## word2vec 모델 저장 및 로드
from gensim.models import KeyedVectors

model.wv.save_word2vec_format('eng_w2v')  # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format('eng_w2v')  # 모델 로드
