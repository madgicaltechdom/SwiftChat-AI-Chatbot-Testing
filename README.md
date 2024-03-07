# Evaluating and Scoring the AI chatbot answer similarity.

We Have different SwiChat AI Chatbots that is designed to asnwer user questions. We have question bank that consist three main columns, "Question", "Expected Answer" and "Chatbot Response". We aim to determine the semantic similarity of "Chatbot Response" with "Expected Answer."

Before moving ahead, create your dataset that contains the Questions bank. Here is a sample sheet with script that you can use for generating answers and as dataset.

Sheet: [Link](https://docs.google.com/spreadsheets/d/1yld4DYDhZsANIj9MjjVdmP5aYpZcBF3QCFamXMlriBQ/edit?usp=sharing)

To conduct this evaluation, we employ two embedding models: Bert and Glove. From each of these models, we calculate two types of matrices:<br/>

- **Recall**: This measures the number of references found in the response divided by the size of the reference list.

- **Precision**: This calculates the number of responses found in the reference list divided by the size of the response list.

Here's how we calculate these scores using **BERT**:

**For Recall**: Initially, both the reference and response data are transformed into lists by splitting them at commas. Next, we compute the cosine similarity for each reference data point and compare it with every response data point. If the similarity exceeds a predetermined threshold, set at 0.95, we consider it a match and label it as "1," storing it in a separate list. This process is repeated for each response, resulting in a list of "1s" and "0s." To calculate the final score, we divide the total number of "1s" (indicating similarity found in the response data) by the length of the reference data.

**For Precision**: Similar to the process for Recall, we transform both the reference and response data into lists. We compute the cosine similarity for each response data point and compare it with every reference data point. If the similarity exceeds the threshold of 0.95, we label it as "1" and store it in a separate list. Again, for each response, this process is repeated, resulting in a list of "1s" and "0s." To calculate the final Precision score, we divide the total number of "1s" (indicating similarity found in the reference data) by the length of the response data.

Now, for the **GloVe** embedding model:

**For Recall**: Similar to the BERT approach for Recall, we calculate the score, but this time utilizing GloVe embeddings. In this case, we use a cosine similarity threshold of 0.87.

**For Precision**: Similarly, like the BERT approach for Precision, we calculate the score using GloVe embeddings with a cosine similarity threshold of 0.87

Now, for the **Bge_large** embedding model:


**For Recall**: Similar to the BERT approach for Recall, we calculate the score with a threshold of 0.87, but this time utilizing Bge_large. The full name of model is BAAI/bge-large-en-v1.5 on hugging face. In this case, we first load the model using sentence transformer then encode the text using model and then making dot product on ecoded data and create score. 

**For Precision**: Similarly, like the BERT approach for Precision, we calculate the score using Bge_large embeddings with threshold of 0.87.

**Bleu**: We can also create Bleu metric with 2-gram(matching two word sequence).<br/>
**Bleurt**: We can also create a Bleurt metric. we use this bleurt model with different checkpoints like some of them `bleurt-large-512`, `BLEURT-20` four more model you can refere this [file](https://github.com/google-research/bleurt/blob/master/checkpoints.md). Here all checkpoint is described..<br/>


## Prerequisite
`pip3 install pandas`<br/>
`pip3 install torch`<br/>
`pip3 install numpy`<br/>
`pip3 install transformers`<br/>
`pip3 install scipy`<br/>
`pip3 install scikit-learn`<br/>
`pip3 install evaluate`<br/>
`pip3 install datasets==2.10.0`<br/>
`pip3 install -U sentence-transformers`<br/>

## How to run 
[Demo Video](https://drive.google.com/file/d/1--g1NrrPmnlEKcHNbudUlFlj7V-Z9arT/view?usp=sharing) for reference.<br/>
**Step 1**<br/>
You need to make a clone of this repo or download this repo directly. For making clone of this repo you can follow this command.
```
     git clone https://github.com/madgicaltechdom/SwiftChat-AI-Chatbot-Testing.git
```
**Step 2**<br/>
Then open the cloned folder in VS Code 

**Step 3**<br/>
Then we need to add the csv file in `data` folder.<br/>

**Step 4**<br/>
Then you need to download the glove file. For downloading of glove file you can follow this [link](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt). After downloading the zip file you need to extract it and paste `glove.6B.100d.txt` file in the `glove_data` folder in same directory where we are working.<br/>
**Step 5**<br/>
Then go to the `main.py` file. Add the reference column name in line 19 and model response column name in line 20, add metric type in line 21.<br/>

**Step 6**<br/>
If you are using Bleurt model then you nedd the first download the desrired checkpoint and save it in the same directory on which we are working.
Like for `BLEURT-20` you can follow this command.
```
    wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
```
```
    unzip BLEURT-20.zip
```
or for `bleurt-large-512` you can follow this command.
```
    wget https://storage.googleapis.com/bleurt-oss/bleurt-large-512.zip
```
```
    unzip bleurt-large-512.zip
```

or you can directly download from this [file](https://github.com/google-research/bleurt/blob/master/checkpoints.md)<br/>

Then you need to define the name of checkpoint in `main.py ` file in line 42

**Step 7**<br/>
Then you need to run this command in terminal.
```
     python main.py
```

## Output
The script generates an output CSV file named `response.csv` with an additional column containing evaluation scores according to our metric.<br/>

The additional columns are as follows:-<br/>
- If we choose `Bert_metric`
1. **Precision scores Bert**:- This column represents the Precision score using BERT.
2. **Recall scores Bert**:- This column represents the Recall score using BERT.
- If we choose `Glove_metric`
1. **Precision scores Glove**:- This column represents the Precision score using the Glove.
2. **Recall scores Glove**:- This column represents the Recall score using Glove.
- If we choose `Bge_large_metric`
1. **Precision scores Bge_large**:- This column represents the Precision score using Bge_large.
2. **Recall scores Bge_large**:- This column represents the Recall score using Bge_large.
- If we choose `Bleu`
1. **Bleu Score**:- This column represents the Bleu score
- If we choose `Bleurt`
1. **Bleurt Score**:- This column represents the Bleurt score




## Reference
I take information from this [article](https://iq.opengenus.org/different-techniques-for-sentence-semantic-similarity-in-nlp/
). For [bge_large embedding](https://huggingface.co/BAAI/bge-large-en-v1.5)https://huggingface.co/BAAI/bge-large-en-v1.5
