import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class bge_large_metric:
    def __init__(self, refernce_data, response_data, model):
        self.refernce_data = refernce_data
        self.response_data = response_data
        self.model = model
    
    def calculate_score(self):
        

        # Tokenize input sentences
        sentences1 = [self.refernce_data]
        sentences2 = [self.response_data]
        embeddings_1 = self.model.encode(sentences1, normalize_embeddings=True)
        embeddings_2 = self.model.encode(sentences2, normalize_embeddings=True)
        similarity = embeddings_1 @ embeddings_2.T
        print(similarity)

        return similarity[0][0]