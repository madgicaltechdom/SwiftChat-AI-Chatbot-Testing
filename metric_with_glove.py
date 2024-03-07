import numpy as np
from scipy.spatial.distance import cosine


class metric_with_glove:
    def __init__(self, refernce_data, response_data,embeddings_index):
        self.refernce_data = refernce_data
        self.response_data = response_data
        self.embeddings_index = embeddings_index
        

    def calculate_score(self):
        sentence1 = self.refernce_data
        sentence2 = self.response_data

        words1 = sentence1.split()
        words2 = sentence2.split()

        embeddings1 = [self.embeddings_index.get(word, np.zeros(100)) for word in words1]
        embeddings2 = [self.embeddings_index.get(word, np.zeros(100)) for word in words2]

        sentence_embedding1 = np.mean(embeddings1, axis=0)
        sentence_embedding2 = np.mean(embeddings2, axis=0)

        # Compute cosine similarity

        similarity = 1 - cosine(sentence_embedding1, sentence_embedding2)
        return similarity
