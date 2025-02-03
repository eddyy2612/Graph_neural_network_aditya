# Classification of Publicly Listed Companies Using GNNs

## Overview
This project aims to classify publicly-listed companies based on their descriptions using a Graph Neural Network (GNN) approach. The methodology is inspired by the research paper *"Graph Convolutional Networks for Text Classification"* by Liang Yao, Chengsheng Mao, and Yuan Luo.

## Approach
The solution is designed using the following steps:

1. **Data Preprocessing and Tokenization**
   - The textual descriptions of companies are processed and tokenized for further use.

2. **Graph Construction**
   - The text data is converted into a graph representation.
   - An adjacency matrix is created using:
     - PMI (Pointwise Mutual Information) for word-word relationships.
     - TF-IDF scores for document-word relationships.
     - Identity values for self-connections.

3. **Modeling with Graph Neural Networks (GNNs)**
   - A GNN-based classifier is used to process the graph representation of the dataset.
   - The model leverages graph convolutional networks (GCNs) to improve classification accuracy.

## Execution Constraints
Due to platform limitations (e.g., Kaggle, Jupyter), execution was not fully completed. However, the approach has been carefully designed by reviewing multiple research papers and should work effectively when run on a virtual machine with sufficient RAM.

## Requirements
To successfully run this project, ensure the following dependencies are installed:
- Python 3.x
- PyTorch
- NetworkX
- SciPy
- NumPy
- Scikit-learn
- Other required libraries as specified in the notebook

## Usage
1. Set up a virtual machine with sufficient RAM.
2. Install the required dependencies.
3. Run the notebook to preprocess the data and train the GNN model.

