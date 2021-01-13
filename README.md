# Exploring Gender Biases in Information Retrieval Relevance Judgement Datasets
This repository contains the code and resources for detecting the gender of queries (Female, Male, Neutral) along with psychological characteristics of their relevance judgement documents. 
## Query Gender Identification and Labeling
In this work, we proposed a Query Gender classifier. As the first step and in order to be able to label queries based on their gender at scale, we employed the [gender-annotated dataset](https://github.com/navid-rekabsaz/GenderBias_IR/blob/master/resources/queries_gender_annotated.csv) released by Navid Rekabsaz and Markus Schedl to train relevant classifiers. This dataset consists of 742 female, 1,202 male and 1,765 neutral queries. We trained various types of  classifiers on this dataset and in order to evaluate the performance of the classifiers, we adopt a 5-fold cross-validation strategy.
<table>
<thead>
  <tr>
    <th style="text-align: right;" class="tg-0lax">Category</th>
    <th class="tg-0lax">Classifier</th>
    <th class="tg-0lax">Accuracy</th>
    <th class="tg-baqh" colspan="3">F1-Score</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax" rowspan="6"><br>Dynamic Embeddings<br></td>
    <td class="tg-0lax">BERT (base uncased)</td>
    <td class="tg-l2oz"><b>0.856</td>
    <td class="tg-l2oz"><b>0.816</td>
    <td class="tg-l2oz"><b>0.872</td>
    <td class="tg-l2oz"><b>0.862</td>
  </tr>
  <tr>
    <td class="tg-0lax">DistilBERT (base uncased)</td>
    <td class="tg-lqy6">0.847</td>
    <td class="tg-lqy6">0.815</td>
    <td class="tg-lqy6">0.861</td>
    <td class="tg-lqy6">0.853</td>
  </tr>
  <tr>
    <td class="tg-0lax">RoBERTa</td>
    <td class="tg-lqy6">0.810</td>
    <td class="tg-lqy6">0.733</td>
    <td class="tg-lqy6">0.820</td>
    <td class="tg-lqy6">0.836</td>
  </tr>
  <tr>
    <td class="tg-0lax">DistilBERT (base cased)</td>
    <td class="tg-lqy6">0.800</td>
    <td class="tg-lqy6">0.730</td>
    <td class="tg-lqy6">0.823</td>
    <td class="tg-lqy6">0.833</td>
  </tr>
  <tr>
    <td class="tg-0lax">BERT (base cased)</td>
    <td class="tg-lqy6">0.797</td>
    <td class="tg-lqy6">0.710</td>
    <td class="tg-lqy6">0.805</td>
    <td class="tg-lqy6">0.827</td>
  </tr>
  <tr>
    <td class="tg-0lax">XLNet (base cased)</td>
    <td class="tg-lqy6">0.795</td>
    <td class="tg-lqy6">0.710</td>
    <td class="tg-lqy6">0.805</td>
    <td class="tg-lqy6">0.826</td>
  </tr>
  <tr>
    <td class="tg-0lax" rowspan="2">Static Embeddings</td>
    <td class="tg-0lax">Word2Vec</td>
    <td class="tg-lqy6">0.757</td>
    <td class="tg-lqy6">0.626</td>
    <td class="tg-lqy6">0.756</td>
    <td class="tg-lqy6">0.809</td>
  </tr>
  <tr>
    <td class="tg-0lax">fastText</td>
    <td class="tg-lqy6">0.750</td>
    <td class="tg-lqy6">0.615</td>
    <td class="tg-lqy6">0.759</td>
    <td class="tg-lqy6">0.792</td>
  </tr>
</tbody>
</table>

In the table above the performance of each of the developed classifiers is reported. As shown the uncased **fine-tuned BERT** model shows the best
performance for query gender identification. Finally, for the purpose of measuring bias in relevance judgements, we used our best-performed model to identify the gender of queries in MS MARCO Dev set that had at least one related human-judged relevance judgement document - equivalent to 51,827 queries. Note that, the queries of gender-annotated dataset were removed from this dataset to avoid unintended leakage.

The following table illustrates a few queries labeled using our fine-tuned BERT classifier. Here is all the the [1405 female](https://github.com/genderbias/gender-bias-in-relevance-judgements/blob/main/results/identified%20gendered%20queries/female_queries.csv), [1405 male](https://github.com/genderbias/gender-bias-in-relevance-judgements/blob/main/results/identified%20gendered%20queries/male_queries.csv), and [1405 neutral](https://github.com/genderbias/gender-bias-in-relevance-judgements/blob/main/results/identified%20gendered%20queries/neutral_queries.csv) labeled queries.
|QID     |Query                                        |Predicted Gender |
|---------|----------------------------------------------|------------------|
|80095   |can you take naproxen during **pregnancy**       |Female           |
|14757   |**aimee osbourne** net worth                     |Female           |
|189154  |foods that can prevent **prostate** cancer       |Male             |
|11251   |**adam devine** net worth                        |Male             |
|40234   |average percentage of accepted scholarships  |Neutral          |

### Code
- **Training** - [`code/train.py`](https://github.com/genderbias/gender-bias-in-relevance-judgements/blob/main/code/train.py): The code for fine-tuning BERT on queries_gender_annotated dataset or any other dataset.
- **Predicting** - [`codes/predict.py`](https://github.com/genderbias/gender-bias-in-relevance-judgements/blob/main/code/predict.py): In any case that you do not want to train the model, you can download our [fine-tuned model](https://drive.google.com/file/d/1_YTRs4v5DVUGUffnRHS_3Yk4qteJKO6w/view?usp=sharing) and use `predict.py`  for predicting the gender of queries.


## Psychological Characteristics Quantification
Our approach for quantifying bias is based on measuring different psychological characteristics of the relevance judgement documents associated with each query. To investigate this, we employ Linguistic Inquiry and Word Count (LIWC) text analytics toolkit to compute the degree to which different psychological characteristics are observed in relevance judgement documents. These Psychological characteristics related to the queries of each group can be found in [`results/psychological analysis`](https://github.com/genderbias/gender-bias-in-relevance-judgements/tree/main/results/psychological%20analysis) folder.

