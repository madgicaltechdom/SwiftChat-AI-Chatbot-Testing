from bert_metric import bert_metric
from metric_with_glove import metric_with_glove
from bge_large import bge_large_metric


class precision:
    def __init__(self, reference_list, response_list, model_type, tokenizer, bert_model, embeddings_index, bge_large_model):
        self.reference_list = reference_list
        self.response_list = response_list
        self.model_type = model_type
        self.full_precision_score_list=[]
        self.embeddings_index = embeddings_index
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.bge_large_model = bge_large_model


    def calaculate_precision(self):


        if self.model_type=="Bert":
            # Comparing each response data with every reference
            for response_data in self.response_list:
                precision_score_list=[]
                response_data = response_data.replace("-"," ")
                response_data = response_data.strip()
                for index, reference_data in enumerate(self.reference_list):
                    reference_data = reference_data.replace("-"," ")
                    reference_data = reference_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = bert_metric(reference_data,response_data,self.tokenizer,self.bert_model)
                        scores = metric.calculate_score()
                        precision_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.95 for item in precision_score_list) else 0
                self.full_precision_score_list.append(result)
            score_precision = (self.full_precision_score_list.count(1)/len(self.response_list))*100
            evaluation_score_precision = round(score_precision,3)

        elif self.model_type=="Glove":
            # Comparing each response data with every reference
            for response_data in self.response_list:
                precision_score_list=[]
                response_data = response_data.replace("-"," ")
                response_data = response_data.strip()
                for index, reference_data in enumerate(self.reference_list):
                    reference_data = reference_data.replace("-"," ")
                    reference_data = reference_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = metric_with_glove(reference_data,response_data,self.embeddings_index)
                        scores =  metric.calculate_score()
                        precision_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.87 for item in precision_score_list) else 0
                self.full_precision_score_list.append(result)
            # print(precision_score_list)
            score_precision = (self.full_precision_score_list.count(1)/len(self.response_list))*100
            evaluation_score_precision = round(score_precision,3)

        elif self.model_type=="Bge_large":
            # Comparing each response data with every reference
            for response_data in self.response_list:
                precision_score_list=[]
                response_data = response_data.replace("-"," ")
                response_data = response_data.strip()
                for index, reference_data in enumerate(self.reference_list):
                    reference_data = reference_data.replace("-"," ")
                    reference_data = reference_data.strip()
                    if reference_data and response_data and "â€" not in response_data and "â€" not in reference_data:
                        metric = bge_large_metric(reference_data,response_data,self.bge_large_model)
                        scores =  metric.calculate_score()
                        precision_score_list.append(scores)
                        # print(response_data,reference_data,scores)
                result = 1 if any(item > 0.87 for item in precision_score_list) else 0
                self.full_precision_score_list.append(result)
            # print(precision_score_list)
            score_precision = (self.full_precision_score_list.count(1)/len(self.response_list))*100
            evaluation_score_precision = round(score_precision,3)

        return evaluation_score_precision, self.full_precision_score_list 
