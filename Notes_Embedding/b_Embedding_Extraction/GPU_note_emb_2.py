# # Install all relevant packages
# !pip install transformers
# !pip install torch
# !pip install pip install mendelai-brat-parser
# !pip install smart-open
# !pip install -U scikit-learn



# ****************************************************************************
# Import libraries
import transformers
import torch
import torch.nn as nn
import itertools

from transformers import BertTokenizer, BertForTokenClassification, BertModel
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizerFast,  BatchEncoding
from tokenizers import Encoding
from transformers import AutoTokenizer, AutoModel


from brat_parser import get_entities_relations_attributes_groups
import zipfile
import os

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass
from typing import List
from torch.utils.data.dataloader import DataLoader
from transformers import BertForTokenClassification, AdamW
from torch.nn import functional as F
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd


# ****************************************************************************
print('Ensure CUDA is available, otherwise use CPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU details:", torch.cuda.get_device_name(0))

# ****************************************************************************
# Read in clinical notes file

file_path = 'MODIFIED_NOTEEVENTS_2083180'
clinical_notes = pd.read_csv(f'../../Data/MIMIC_resources/{file_path}.csv')

# ****************************************************************************
# Text column is the section of interest
# Creating a subsampled dataframe as an example with the first 1000 notes

subsampled_notes_df = clinical_notes
print(subsampled_notes_df.shape)
clin_notes = subsampled_notes_df['TEXT'].tolist()

# *************************************************************************
def tokenize_batch(notes_slice):
    return tokenizer(notes_slice, padding=True, truncation=True, max_length=512, return_tensors="pt")


# *************************************************************************
batch_size = 16  # Adjust based on your GPU memory

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

# Move the model to the selected device (GPU or CPU)
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)

embeddings = []

with torch.no_grad():
    # Process the dataset in chunks
    for i in range(0, len(clin_notes), batch_size):
        notes_slice = clin_notes[i:i+batch_size]
        inputs = tokenize_batch(notes_slice)

        # Move the tokenized inputs to the GPU
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        
        hidden_states = outputs.last_hidden_state
        cls_embeddings = hidden_states[:, 0, :].to('cpu').numpy()
        embeddings.append(cls_embeddings)

# embeddings is a list of arrays, one per batch. You may want to concatenate them.
embeddings = np.concatenate(embeddings, axis=0)

embeddings_flattened = [embedding.flatten() for embedding in embeddings]

data = {"Clinical Note": subsampled_notes_df['ROW_ID'].tolist(), "Embedding": embeddings_flattened}

pd.DataFrame(data).to_csv(f'../../Data/MIMIC_resources/emb_{file_path}.csv', index=False)


