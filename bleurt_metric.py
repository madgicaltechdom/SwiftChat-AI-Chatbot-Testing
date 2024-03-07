from bleurt import score

class bleurt_metric:
    def __init__(self, refernce_data, response_data, checkpoint_model):
        self.refernce_data = [refernce_data]
        self.response_data = [response_data]
        self.checkpoint_model = checkpoint_model

    def bleurt(self):
            scores = self.checkpoint_model.score(references=self.refernce_data, candidates=self.response_data)
            print(scores)
            rounded_off_score = round(scores[0], 3)
            return rounded_off_score
