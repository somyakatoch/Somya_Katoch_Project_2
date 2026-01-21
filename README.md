## Model Overview

The web app uses an **XGBoost (Extreme Gradient Boosting) classifier** to predict credit risk.  

- **Type:** Supervised Machine Learning (Classification)  
- **Input:** Customer features such as income, credit history, loan amount, etc.  
- **Output:** Probability of being **low risk** or **high risk**  
- **Training:** Model is trained on a labeled dataset and optimized using **RandomizedSearchCV** to find the best hyperparameters for maximum accuracy and AUROC.  
- **Performance:** The tuned model provides reliable predictions and is robust to overfitting due to XGBoost’s regularization features.  

This ensures that the app can give **real-time, accurate predictions** based on user-provided data.
