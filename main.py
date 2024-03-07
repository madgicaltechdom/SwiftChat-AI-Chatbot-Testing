import pandas as pd
import numpy as np
from bert_metric import bert_metric
from bleu_metric import bleu_metric
from transformers import BertTokenizer, BertModel
from precision import precision
from recall import recall
from bleurt_metric import bleurt_metric
from sentence_transformers import SentenceTransformer
from bleurt import score
import os

files = os.listdir("data")

if len(files) == 1:
    file_name = os.path.join("data", files[0])
    df = pd.read_csv(file_name)
    
# Defining reference data and response data from csv
reference = df["Expected Answer"]
response = df["Chatbot Response"]
metric_type = ["Bert_metric"]

# Load tokenizer and Bert model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

#Loade bge_large
bge_large_model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Load pre-trained GloVe word embeddings
embeddings_index = {}
with open('../glove_data/archive1/glove.6B.100d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Define checkpoint of bleurt you want to use
checkpoint = "BLEURT-20"
bleurt_checkpoint_model = score.BleurtScorer(checkpoint)

overall_recall_scores_bert=[]
overall_precision_scores_bert=[]
evaluation_score_list_recall_bert = []
evaluation_score_list_precision_bert = []
overall_recall_scores_bge_large=[]
overall_precision_scores_bge_large=[]
evaluation_score_list_recall_bge_large = []
evaluation_score_list_precision_bge_large = []
overall_recall_scores_glove=[]
overall_precision_scores_glove=[]
evaluation_score_list_recall_glove = []
evaluation_score_list_precision_glove = []
blert_score_list =[]
bleu_score_list= []
i=0
for resp, ref in zip(response, reference):
    print(resp)
    response_list = resp.split(",")
    reference_list = ref.split(",")

    
    if "Bert_metric" in metric_type:
        # Here we calaculate precision score "Number of responses found in reference/size of response list"
        bert_precision = precision(reference_list,response_list,"Bert",tokenizer,bert_model,embeddings_index,bge_large_model)
        bert_recall = recall(reference_list,response_list,"Bert",tokenizer,bert_model,embeddings_index,bge_large_model)
        score_bert_precision, score_bert_list_precision = bert_precision.calaculate_precision()
        recall_score_bert, recall_score_bert_list = bert_recall.calaculate_recall()
        evaluation_score_list_precision_bert.append(score_bert_list_precision)
        overall_precision_scores_bert.append(score_bert_precision)
        evaluation_score_list_recall_bert.append(recall_score_bert_list)
        overall_recall_scores_bert.append(recall_score_bert)
        

    if "Glove_metric" in metric_type:
        # Here we calaculate recall score "Number of references found in Response / size of reference list"
        glove_precision = precision(reference_list,response_list,"Glove",tokenizer,bert_model,embeddings_index,bge_large_model)
        glove_recall = recall(reference_list,response_list,"Glove",tokenizer,bert_model,embeddings_index,bge_large_model)
        score_glove_precision, score_glove_list_precision = glove_precision.calaculate_precision()
        recall_score_glove, recall_score_glove_list = glove_recall.calaculate_recall()  
        evaluation_score_list_recall_glove.append(recall_score_glove_list)
        overall_recall_scores_glove.append(recall_score_glove)
        evaluation_score_list_precision_glove.append(score_glove_list_precision)
        overall_precision_scores_glove.append(score_glove_precision)


    if "Bge_large_metric" in metric_type:
        # Here we calaculate recall score "Number of references found in Response / size of reference list"
        glove_precision = precision(reference_list,response_list,"Bge_large",tokenizer,bert_model,embeddings_index,bge_large_model)
        glove_recall = recall(reference_list,response_list,"Bge_large",tokenizer,bert_model,embeddings_index,bge_large_model)
        score_bge_large_precision, score_bge_large_list_precision = glove_precision.calaculate_precision()
        recall_score_bge_large, recall_score_bge_large_list = glove_recall.calaculate_recall()  
        evaluation_score_list_recall_bge_large.append(recall_score_bge_large_list)
        overall_recall_scores_bge_large.append(recall_score_bge_large)
        evaluation_score_list_precision_bge_large.append(score_bge_large_list_precision)
        overall_precision_scores_bge_large.append(score_bge_large_precision)
    
    if "Bleurt" in metric_type:
        # Here we calculate blert score by string matching
        call_bleurt_metric = bleurt_metric(ref,resp,bleurt_checkpoint_model)
        bleurt_score = call_bleurt_metric.bleurt()
        blert_score_list.append(bleurt_score)
    
    if "Bleu" in metric_type:
        call_bleu_metric = bleu_metric(ref,resp)
        bleu_score = call_bleu_metric.bleu()
        bleu_score_list.append(bleu_score)

    i = i+1
    print(i)

if "Bert_metric" in metric_type:
    df["Precision scores Bert"] = overall_precision_scores_bert
    df["Recall scores Bert"] = overall_recall_scores_bert
    # df["Matching list on precision for Bert"] = evaluation_score_list_precision_bert
    # df["Matching list on recall for Bert"] = evaluation_score_list_recall_bert

if "Glove_metric" in metric_type:
    df["Precision scores Glove"] = overall_precision_scores_glove
    df["Recall scores Glove"] = overall_recall_scores_glove
    # df["Matching list on precision for Glove"] = evaluation_score_list_precision_glove
    # df["Matching list on recall for Glove"] = evaluation_score_list_recall_glove

if "Bge_large_metric" in metric_type:
    df["Precision scores Bge_large"] = overall_precision_scores_bge_large
    df["Recall scores Bge_large"] = overall_recall_scores_bge_large
    # df["Matching list on precision for Glove"] = evaluation_score_list_precision_glove
    # df["Matching list on recall for Glove"] = evaluation_score_list_recall_glove

if "Bleurt" in metric_type:
    df["Bleurt Score"] = blert_score_list

if "Bleu" in metric_type:
    df["Bleu Score"] = bleu_score_list



df.to_csv('response1.csv',index=False)
