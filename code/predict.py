# importing libraries
import transformers
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Constant variables 
class_names = ['Female', 'Male' , 'Neutral']
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
TEST_BATCH_SIZE = 16
MAX_LEN = 55

# Dataset
class GenderBiasDataset(Dataset):

    def __init__(self, queries, tokenizer, max_len):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_text = str(self.queries[index])
         
        encoding = self.tokenizer.encode_plus(
            query_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        return {
                'query': query_text,
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }

# Dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GenderBiasDataset(
    queries = df['query'].to_numpy(),
    tokenizer  =tokenizer,
    max_len = max_len
  )
  return DataLoader(
    ds,
    batch_size = batch_size,
    num_workers = 5 # num_workers should be 0 while running the code on CPU
  )

#Prediction function
def get_predictions(model, data_loader):
  model = model.eval()
  query_texts = []
  predictions = []
  prediction_probs = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["query"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs[0], dim=1)
      query_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs[0])
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  return query_texts, predictions, prediction_probs


#Reading MSMarco dev set queries (these queires do not have label)
df = pd.read_csv("msmarco.csv") # a dataframe containing the queries
test_data_loader = create_data_loader(df, tokenizer, MAX_LEN, TEST_BATCH_SIZE)

#Loading the fine-tuned model - you can download the model from https://drive.google.com/file/d/1_YTRs4v5DVUGUffnRHS_3Yk4qteJKO6w/view?usp=sharing
print("Loading the Model")
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 3)
model.load_state_dict(torch.load("BERT_fine_tuned.bin", map_location = device))
print("Model Loaded Successfully")

print("Prediction started")
y_query_texts, y_pred, y_pred_probs = get_predictions(model, test_data_loader)
prediction = pd.DataFrame(df.values.tolist(), columns = ["qid","query"])
prediction['female_probability'] = y_pred_probs[:, 0]
prediction['male_probability'] = y_pred_probs[:, 1]
prediction['neutral_probability'] = y_pred_probs[:, 2]
prediction['prediction'] = y_pred
prediction.to_csv("predictions.csv", index = False)
