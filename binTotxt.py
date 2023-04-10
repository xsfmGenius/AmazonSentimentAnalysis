# # 词向量格式转换
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
model.save_word2vec_format('GoogleNews-vectors-negative300.txt', binary=False)
# print(len(model))
# print(model.vector_size)
# print(model["like"])