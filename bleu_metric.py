from nltk.translate.bleu_score import sentence_bleu


class bleu_metric:
    def __init__(self, refernce_data, response_data):
        self.refernce_data = [refernce_data.split()]
        self.response_data = response_data.split()
    def bleu(self):
            
        scores = sentence_bleu( self.refernce_data ,  self.response_data ,weights=(0, 1, 0, 0))
        rounded_off_score = round(scores,3)
                
        return rounded_off_score