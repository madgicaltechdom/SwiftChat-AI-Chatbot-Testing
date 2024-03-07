import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


class bert_metric:
    def __init__(self, refernce_data, response_data,tokenizer, model):
        self.refernce_data = refernce_data
        self.response_data = response_data
        self.tokenizer = tokenizer
        self.model = model
    
    def calculate_score(self):
        

        # Tokenize input sentences
        sentences1 = [self.refernce_data]
        sentences2 = [self.response_data]
        encoded_inputs1 = self.tokenizer(sentences1, padding=True, truncation=True, return_tensors='pt')
        encoded_inputs2 = self.tokenizer(sentences2, padding=True, truncation=True, return_tensors='pt')

        # Compute BERT embeddings for input sentences
        with torch.no_grad():
            outputs1 = self.model(**encoded_inputs1)
            outputs2 = self.model(**encoded_inputs2)

        # Get last layer hidden states as sentence embeddings
        embeddings1 = outputs1.last_hidden_state[:, 0, :]
        embeddings2 = outputs2.last_hidden_state[:, 0, :]

        # Compute cosine similarity between two sentence embeddings
        similarity = cosine_similarity(embeddings1[0].unsqueeze(0), embeddings2[0].unsqueeze(0))

        return similarity[0][0]
