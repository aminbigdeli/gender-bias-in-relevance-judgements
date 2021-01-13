# importing libraries
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
import pandas as pd
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

#Defining some key variables for preprocessing step
class_names = ['Female', 'Male' , 'Neutral']
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 33
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 2e-5
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#Dataset
class GenderBiasDataset(Dataset):

    def __init__(self, queries, targets, tokenizer, max_len):
        self.queries = queries
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query_text = str(self.queries[index])
        target = self.targets[index]
         
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
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'targets': torch.tensor(target, dtype=torch.long)
        }
        
#Dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GenderBiasDataset(
    queries = df['query'].to_numpy(),
    targets = df['label'].to_numpy(),
    tokenizer  =tokenizer,
    max_len = max_len
  )
  return DataLoader(
    ds,
    batch_size = batch_size,
    num_workers = 5
  )

#Training function
def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      labels = targets
    )
    _, preds = torch.max(outputs[1], dim=1)  # the second return value is logits
    loss = outputs[0] #the first return value is loss
    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  return correct_predictions.double() / n_examples, np.mean(losses)

#Evaluation function - used when adopting K-fold
def eval_model(model, data_loader, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      loss = outputs[0]
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses)

#Prediction function - used to calculate the accuracy of the model when true labels are available
def get_predictions(model, data_loader):
  model = model.eval()
  query_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["query"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
	labels = targets
      )
      _, preds = torch.max(outputs[1], dim=1)
      query_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs[1])
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return query_texts, predictions, prediction_probs, real_values


#Fine-Tuning the BERT on the Dataset
result = open("BERT_Tuninig_results.txt", "w")
df = pd.read_csv("queries_gender_annotated.csv", names = ["query", "label"]) 
labelEncoder = LabelEncoder()
df['label'] = labelEncoder.fit_transform(df['label'])
result.write("Shape of Dataset: {} \n".format(df.shape))
wordlist = pd.read_csv("gender_specific_wordlist.csv")
wordlist['label'] = labelEncoder.fit_transform(wordlist['label'])
df = pd.concat([df, wordlist], ignore_index = False)
result.write("Shape of Dataset after concatination with wordlist: {} \n".format(df.shape))

train_data_loader = create_data_loader(df, tokenizer, MAX_LEN, TRAIN_BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 3) 
model = model.to(device)

optimizer = AdamW(params =  model.parameters(), lr = LEARNING_RATE, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = total_steps
        )

for epoch in range(EPOCHS):
    result.write(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    result.write("\n")
    result.write('-' * 10)
    result.write("\n")
    train_acc, train_loss = train_epoch(
                model,
                train_data_loader,
                optimizer,
                device,
                scheduler,
                len(df)
        )
    result.write(f'Train loss {train_loss} accuracy {train_acc}')
    result.write("\n")

torch.save(model.state_dict(), "BERT_fine_tuned.bin")
result.close()
