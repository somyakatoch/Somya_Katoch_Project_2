# Credit Risk Prediction Web App

This is a **Streamlit web application** that predicts **credit risk** using an **XGBoost classifier**. Users can input customer data through the web interface and get real-time predictions of whether a customer is **low risk** or **high risk**.

---

## Features

- Real-time credit risk prediction.
- Interactive web interface using **Streamlit**.
- Optimized **XGBoost model** tuned with **RandomizedSearchCV**.
- Easy to deploy and run locally.

---

## Model Overview

- **Type:** XGBoost Classifier (Supervised ML)  
- **Input:** Customer features (income, credit history, loan amount, etc.)  
- **Output:** Probability of being low or high risk  
- **Training:** Dataset labeled for credit risk, optimized with **RandomizedSearchCV** for best hyperparameters  
- **Performance:** Accurate and robust predictions, AUROC used for evaluation  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-risk-streamlit.git
cd credit-risk-streamlit
