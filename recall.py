from bert_metric import bert_metric
from metric_with_glove import metric_with_glove
from bge_large import bge_large_metric


class recall:
    def __init__(self, reference_list, response_list, model_type, tokenizer, bert_model, embeddings_index,bge_large_model):
        self.reference_list = reference_list
        self.response_list = response_list
        self.model_type = model_type
        self.full_recall_score_list=[]
        self.embeddings_index = embeddings_index
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.bge_large_model = bge_large_model

    def calaculate_recall(self):

        if self.model_type=="Bert":
            # Comparing each response data with every response
            for reference_data in self.reference_list:
                recall_score_list=[]
                reference_data = reference_data.replace("-"," ")
                reference_data = reference_data.strip()
                for index, response_data in enumerate(self.response_list):
                    response_data = response_data.replace("-"," ")
                    response_data = response_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = bert_metric(reference_data,response_data,self.tokenizer,self.bert_model)
                        scores = metric.calculate_score()
                        recall_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.95 for item in recall_score_list) else 0
                self.full_recall_score_list.append(result)
            score_recall = (self.full_recall_score_list.count(1)/len(self.reference_list))*100
            evaluation_score_recall = round(score_recall,3)

        elif self.model_type=="Glove":
            # Comparing each reference data with every response
            for reference_data in self.reference_list:
                recall_score_list=[]
                reference_data = reference_data.replace("-"," ")
                reference_data = reference_data.strip()
                for index, response_data in enumerate(self.response_list):
                    response_data = response_data.replace("-"," ")
                    response_data = response_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = metric_with_glove(reference_data,response_data,self.embeddings_index)
                        scores =  metric.calculate_score()
                        recall_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.87 for item in recall_score_list) else 0
                self.full_recall_score_list.append(result)
            score_recall = (self.full_recall_score_list.count(1)/len(self.reference_list))*100
            evaluation_score_recall = round(score_recall,3)


        elif self.model_type=="Bge_large":
            # Comparing each reference data with every response
            for reference_data in self.reference_list:
                recall_score_list=[]
                reference_data = reference_data.replace("-"," ")
                reference_data = reference_data.strip()
                for index, response_data in enumerate(self.response_list):
                    response_data = response_data.replace("-"," ")
                    response_data = response_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = bge_large_metric(reference_data,response_data,self.bge_large_model)
                        scores =  metric.calculate_score()
                        recall_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.87 for item in recall_score_list) else 0
                self.full_recall_score_list.append(result)
            score_recall = (self.full_recall_score_list.count(1)/len(self.reference_list))*100
            evaluation_score_recall = round(score_recall,3)

        return evaluation_score_recall, self.full_recall_score_list 
