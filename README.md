# Amazon Recommender System with Graph Neural Networks (GNNs)

Welcome to the **Amazon Recommender System** project! This repository implements a recommendation system for Amazon products using **Graph Neural Networks (GNNs)**. The system goes beyond traditional recommendation algorithms like collaborative filtering by leveraging advanced graph-based methods to uncover deeper insights into user-product interactions.

---

## 📌 Overview

### Motivation
Recommendation systems play a vital role in personalizing user experiences. Amazon's ability to recommend products effectively relies on advanced algorithms. This project explores **Graph Neural Networks** to simulate such a system.

### Highlights
- Built using the **Amazon 2018 dataset** with a subset processed using the **5-kcore** method.
- Utilizes **GraphSAGE** and **Graph Attention Networks (GAT)** for node embeddings and aggregation.
- Benchmarked against traditional models like **KNN** and **SVD**.

---

## 🛠️ Features

1. **Graph Representation**:
   - **Nodes**: Users and products.
   - **Edges**: 
     - `Reviews`: Connect users to products with ratings.
     - `Also Bought`: Connect products frequently purchased together.

2. **Node Embeddings**:
   - **Product Nodes**: Generated using `sentence-transformer/all-MiniLM-L6-v2` for descriptions and one-hot encoded categories.
   - **User Nodes**: Initialized as blank and updated dynamically based on interactions.

3. **GNN Models**:
   - **GraphSAGE**: Learns inductive representations of nodes.
   - **GAT**: Assigns attention weights to neighbors for better aggregation.

4. **Explainability**:
   - Integrated gradients via **Captum** to interpret model predictions.

5. **Evaluation**:
   - **Metric**: Root Mean Squared Error (RMSE).
   - Comparison against traditional methods and a state-of-the-art Graph-based Hybrid Recommendation System (GHRS).

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9
- PyTorch Geometric
- Hugging Face Transformers
- Other Python libraries (see `requirements.txt`)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aiden200/ARS.git
   cd ARS

2. Install the required Packages:
  `pip install -r requirements.txt`.
3. Run the Cells in `ARS.ipynb` 


The data is already downloaded in the repository, so there is no need to uncomment the data collection line within the notebook.

Our paper:
[Final_paper](ML567___Final_Project.pdf)
