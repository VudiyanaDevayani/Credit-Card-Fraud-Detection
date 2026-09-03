📌 Credit Card Fraud Detection — Interview One-Page Summary
🔹 Project Title

Hybrid Machine Learning-Based Multi-Stage Framework for Detection of Credit Card Anomalies and Fraud

🔹 Objective

The objective of the project is to detect whether a credit card transaction is legitimate or potentially fraudulent using machine learning.

The main challenge is that fraudulent transactions are very rare compared with legitimate transactions, so simply achieving high accuracy is not enough.

🔹 Dataset

We used the European Credit Card Fraud Dataset.

Total transactions: 284,807
Fraudulent transactions: 492
Fraud percentage: approximately 0.172%
Class 0: Legitimate transaction
Class 1: Fraudulent transaction
Time: Time elapsed from the first transaction
Amount: Transaction amount
V1–V28: PCA-transformed/anonymized features
Why is it difficult?

Because the dataset is highly imbalanced.

284,807 total transactions
        ↓
492 fraud
        ↓
Fraud is extremely rare

A model could achieve high accuracy simply by predicting almost everything as legitimate.

🔹 Project Workflow
Transaction Data
       ↓
Data Preprocessing
       ↓
Feature Extraction
       ↓
Autoencoder
       ↓
Graph Construction
       ↓
GNN
       ↓
LightGBM
       ↓
Fraud Probability
       ↓
Fraud / Legitimate
       ↓
Store Prediction

The project describes preprocessing followed by Autoencoder-based feature extraction, graph-based relationship learning with GNN, and LightGBM as the final classifier.

🔹 1. Data Preprocessing

We prepare the raw data before giving it to the model.

Main steps:

Data cleaning
Handling invalid/missing values
Encoding categorical information where required
Normalizing numerical features
Feature engineering
Train/test splitting

Simple explanation:
Raw data is like raw vegetables. We clean and prepare them before cooking. Similarly, we prepare data before training the model.

🔹 2. Autoencoder

An Autoencoder is used to learn a compact representation of the transaction data.

Original Features
       ↓
    Encoder
       ↓
Compressed Features
       ↓
    Decoder
       ↓
Reconstructed Features

The difference between the original and reconstructed data gives reconstruction error, which can provide information about unusual transactions.

Simple explanation:
It is like making a short summary of a long chapter and then trying to recreate the chapter from that summary.

The implementation contains an encoder and decoder and calculates reconstruction-related information.

🔹 3. Graph Neural Network (GNN)

A GNN is used to understand relationships between entities.

For example:

Account A
    ↓
 Device X
    ↓
Merchant Y
    ↓
Account B

In the graph:

Nodes → accounts, users, devices, merchants, transactions
Edges → relationships between them

The GNN uses message passing to learn information from neighboring/connected nodes.

Why GNN?

Because fraud may not be visible by looking at one transaction alone. Suspicious relationships between accounts, devices or merchants may reveal coordinated fraud.

🔹 4. LightGBM

LightGBM is the final classification model.

It receives:

Transaction Features
       +
Autoencoder Features
       +
GNN Features
       ↓
    LightGBM
       ↓
Fraud Probability

It uses gradient-boosted decision trees to perform the final classification.

Simple explanation:
Autoencoder extracts useful information, GNN understands relationships, and LightGBM makes the final decision.

🔹 Final Output

The system produces:

Fraud probability
Fraud / Legitimate classification

The implementation stores the fraud probability and prediction information in the database.

The current implementation uses:

Probability > 0.5 → Fraud
Probability ≤ 0.5 → Legitimate

🔹 Evaluation

Important metrics:

Accuracy

Percentage of total predictions that are correct.

Precision

Out of the transactions predicted as fraud, how many were actually fraud?

Recall ⭐

Out of all actual fraud transactions, how many did we detect?

F1-score

Balances precision and recall.

Why recall is important?

Because a false negative means:

Actual → FRAUD
Model  → LEGITIMATE

The fraud goes undetected.

For fraud detection, reducing these missed fraud cases is particularly important.

🔹 Web Application

The project also provides a web interface where users can:

Register/login
Enter transaction details
Get fraud predictions
View prediction history

The implementation uses Flask and SQLite for the web application/database functionality shown in the code.
