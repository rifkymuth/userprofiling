import tensorflow as tf
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from torch import cuda

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-05
N_CLASSES = 3

# class Triage(Dataset):
#     def __init__(self, dataframe, tokenizer, max_len):
#         self.len = len(dataframe)
#         self.tweets = dataframe["tweet"].to_numpy()
#         self.sentiment = dataframe["sentimen"].to_numpy()
#         self.tokenizer = tokenizer
#         self.max_len = max_len
        
#     def __getitem__(self, item):
#         tweets = str(self.tweets[item])
#         sentiment = self.sentiment[item]
#         encoding = self.tokenizer.encode_plus(
#         tweets,
#         add_special_tokens=True,
#         max_length=self.max_len,
#         return_token_type_ids=False,
#         pad_to_max_length=True,
#         truncation=True,
#         return_attention_mask=True,
#         return_tensors='pt')
#         return {
#         'tweet_text': tweets,
#          'ids': encoding['input_ids'].flatten(),
#          'mask': encoding['attention_mask'].flatten(),
#          'targets': torch.tensor(sentiment, dtype=torch.long)
#           }
    
#     def __len__(self):
#         return self.len

class IndoBERTModel():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        self.device = 'cuda' if cuda.is_available() else 'cpu'


    def load_model_dict(self, model_path):
        self.model = IndoBERTClass()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def load_model(self, model_path):
        self.model = torch.load(model_path)
        self.model.eval()

    def input_loader(self, batch_input):
        input_set = []
        for review_text in batch_input:
            encoded_review = self.tokenizer.encode_plus(review_text, max_length=MAX_LEN,add_special_tokens=True,
                                                    return_token_type_ids=False,pad_to_max_length=True,return_attention_mask=True,
                                                    return_tensors='pt')
            input_set.append(encoded_review)
        return input_set

    def inference_seq(self, input_set):
        predicted_list = []
        for _,data in enumerate(input_set):
                ids = data['input_ids'].to(self.device, dtype = torch.long)
                mask = data['attention_mask'].to(self.device, dtype = torch.long)

                output = self.model(ids, mask)
                _, prediction = torch.max(output, dim=1)
                predicted_list.append(int(prediction))
        return predicted_list

    def predict(self, tweet_text):
        input_set = self.input_loader(tweet_text) # already raw twitter text format not df
        predicted = self.inference_seq(input_set)

        return predicted

        # predicted_df = pd.Series(data=predicted)
        # df["sentimen"] = predicted_df.sub(1)


class IndoBERTClass(torch.nn.Module):
    def __init__(self):
        super(IndoBERTClass, self).__init__()
        self.indoBERT = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.indoBERT.config.hidden_size, 3)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.indoBERT(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=False
    )
        output = self.drop(pooled_output)
        return self.out(output)

    