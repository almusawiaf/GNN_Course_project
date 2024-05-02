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
clinical_notes = pd.read_csv("../Data/MIMIC_resources/NOTEEVENTS.csv")


# ****************************************************************************
# Text column is the section of interest
# Creating a subsampled dataframe as an example with the first 1000 notes

subsampled_notes_df = clinical_notes.head(10000)
print(subsampled_notes_df.shape)

clin_notes = subsampled_notes_df['TEXT'].tolist()


# *************************************************************************
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")


# *************************************************************************
# Move the model to the selected device (GPU or CPU)
model.to(device)

# *************************************************************************
embeddings = []
counter = 0


# *************************************************************************

with torch.no_grad():
    for note in clin_notes:
        counter += 1
        # Tokenize the note, ensuring to specify padding, truncation, and max length
        inputs = tokenizer(note, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move the inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        outputs = model(**inputs)
        
        # Get the last layer's hidden states
        hidden_states = outputs.last_hidden_state
        
        # Get the embedding for the [CLS] token (first token)
        cls_embedding = hidden_states[:, 0, :]  # Select the first token ([CLS])
        
        # Move the embeddings back to CPU for further processing or storage
        cls_embedding_cpu = cls_embedding.squeeze().to('cpu').tolist()
        
        embeddings.append(cls_embedding_cpu)




# Create a DataFrame containing the clinical notes and their embeddings
data = {"Clinical Note": subsampled_notes_df['ROW_ID'].tolist(), "Embedding": embeddings}


embeddings_df = pd.DataFrame(data)




print(embeddings_df.head())
print(embeddings_df.shape)