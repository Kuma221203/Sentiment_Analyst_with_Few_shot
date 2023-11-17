from datasets import load_dataset
from underthesea import word_tokenize, pos_tag
from transformers import AutoModel, AutoTokenizer
import torch
import re
import numpy as np
import pandas as pd

#get data
dataset = load_dataset("uit-nlp/vietnamese_students_feedback")

data_train = dataset['train'].to_pandas()
data_validation = dataset['validation'].to_pandas()
data_test = dataset['test'].to_pandas()

data_train = pd.concat([data_train, data_validation])

#load tokenizer
phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

def standardize_data(row):
    row = re.sub(r"[\.,\?]+$-", "", row)
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

padded = []
MAX_LEN_PAD = 150

def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

def extract_features(data):
    padded = []

    for i, line in enumerate(data['sentence']):
        line = standardize_data(line)
        line = word_tokenize(line, format="text")
        line = tokenizer.encode(line)
        v_padded = line + [1] * (MAX_LEN_PAD - len(line))
        padded.append(v_padded)

    padded = np.array(padded)
    attention_mask = np.where(padded == 1, 0, 1)
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)
    labels = np.array(data["sentiment"])

    features = []
    steps = labels.shape[0] // 1000

    for i in range(steps):
        print(i)
        _padded = padded[i * 1000:(i + 1)*1000, :]
        _attention_mask = attention_mask[i * 1000:(i + 1)*1000, :]
        with torch.no_grad():
            last_hidden_states = phobert(input_ids= _padded, attention_mask=_attention_mask)

        v_features = last_hidden_states[0][:, 0, :]
        features.append(v_features)
    
    _padded = padded[steps * 1000:, :]
    _attention_mask = attention_mask[steps * 1000:, :]
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= _padded, attention_mask=_attention_mask)

    v_features = last_hidden_states[0][:, 0, :]
    features.append(v_features)

    return torch.cat(features, 0), labels

# Extract train features
X_train, y_train = extract_features(data_train)
np.save('data/train/X_train.npy')
np.save('data/train/y_train.npy')

# Extract test featrure
X_test, y_test = extract_features(data_test)
np.save('data/test/X_test.npy')
np.save('data/test/y_test.npy')

