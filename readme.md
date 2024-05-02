# Enhanced Diagnostic Prediction Using Clinical Data: A Graph Neural Networks Approach

This project develops advanced Graph Neural Network (GNN) models, specifically Graph Convolutional Network (GCN) and GraphSAGE, to predict diagnostics using the MIMIC-III dataset. This dataset, known for its complexity due to high heterogeneity and dimensionality, contains detailed clinical data from patients who have stayed in critical care units. By constructing a heterogeneous graph and applying a meta-path-based approach, the project effectively models the intricate relationships among various types of medical data, including patient information, diagnoses, medications, and procedures.

The primary audience for this project includes researchers and practitioners in the fields of medical informatics and computational health sciences. Specifically, it caters to those interested in applying machine learning and graph theory to improve diagnostic predictions and patient outcome assessments within healthcare datasets. The models and methodologies developed can aid in better understanding patient similarities and health trajectories, which are crucial for personalized medicine and efficient healthcare delivery.

## Installation

Several models have been used within the project. Please create a virtual environment and install the following libraries:
1. torch, pytorch geometric, spacy, scispacy, transformers, 
2. Pandas, numpy, networkx, matplotlib, etc.

## Usage

The project has two major programming section:
### 1. Notes Embedding
    In this folder, you shall find several folders referring to the sequence of steps to be run to extract the final embedding out of the clinical notes given in the MIMIC-III dataset.

#### a. Diagnoses Removal
    Aim: identify diagnoses within the text and replace them with 'X' character. Several models are provided. You can target z_spaCy_NER.ipynb

#### b. Embedding Extraction
    Aim: Use transformers to convert the clinical text to embedding representation.
    You can target (clinical_note_embeddings.ipynb)

#### c. Grouping by Patients
    Aim: Herein, we group the several embedding of one patients into one. We consider a normalized version. 

#### d. Adding embedding for other nodes
    Aim: herein, we add two more numerical representation to the last embedding, to add the other nodes and to distinguish them from each other.

### 2. The main pipeline

#### a. Heterogeneous graph generation
AIM: Read the MIMIC-III dataset and generate the final complete heteogeneours graph.

#### b. Generating X
AIM: Read and process the features of the nodes.

#### c. Generating Y 
AIM: Read and process the labels of the patients, and other nodes.

#### d. Meta-path based similarity
AIM: Read the heterogeneous graph and create k-meta-path based similarity matrices; normalizing each matrix and fusing them by taking the average.

#### e. Downsize the data
AIM: Use PCA model to reduce the dimension of the embedding to the given size. 

#### f. Patient-Level Diagnosis Prediction Using GNN Models
AIM: Given the generated data, this file assemble all the data and feed them to the GNN model (GCN or GraphSAGE). You can also use the bash file to request the services of remote server.


## Acknowledgement

Special thanks are extended to Musaddiq Lodi for his contributions to the diagnosis removal process and the conversion of clinical notes into embedding representations.
Also, sincere thanks extended to dr. Pratip Rana for his assistance and for Professor Thang N. Dinh for offering this significant course.