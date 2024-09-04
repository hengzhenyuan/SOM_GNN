# Graph-based Anomaly Detection

This repository contains code for graph-based anomaly detection.

## Project Structure

- **GT_*.py**: These scripts are used to generate corresponding graph data from various datasets. For example, `GT_SKAB_graphdata.py` processes the SKAB dataset into a graph format suitable for the model.
  
- **model.py**: This script defines the architecture of our Relational Graph Convolutional Network (RGCN) model. The model is designed to handle graph data and includes two RGCN layers followed by ReLU and Softmax activation functions.

- **test_*.py**: These scripts contain the code for training and testing the model on the generated graph data. For example, `test_skabgraph.py` trains and evaluates the model using the graph data generated from the SKAB dataset.

## How to Run

Below are the steps to run the code using the SKAB dataset:

1. **Generate Graph Data**: 
   - First, you need to generate the graph data from the SKAB dataset. This is done by running the following script:
     ```bash
     python GT_SKAB_graphdata.py
     ```
   - This script will process the raw SKAB data and output a graph format that can be used by the model.

2. **Train and Test the Model**:
   - Once the graph data is generated, you can train and test the model by running:
     ```bash
     python test_skabgraph.py
     ```
   - This script will train the RGCN model on the SKAB graph data and evaluate its performance.
