# Churn-Prediction
Machine Learning for Telecom Customers Churn Prediction. will train several classification algorithms to predict the churn rate of telecommunication customers. Machine learning can help companies analyze customer churn rate based on various factors such as services subscribed by customers, tenure rate, and payment method. Predicting churn rate is crucial for these companies as it allows them to focus on retaining existing customers, which is more cost-effective than acquiring new ones.

# Dataset 
Telco Churn Prediction Dataset from kaggle - https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Data Features: The dataset likely contains features such as customer demographics (age, gender, etc.), services subscribed (phone, internet, etc.), payment methods, contract details, and usage patterns. Each of these features may have an impact on customer churn.

Target Variable: The target variable in this dataset is the "Churn" column, which indicates whether a customer churned or not. This is typically a binary variable, with "Yes" indicating churn and "No" indicating no churn.

Class Imbalance: It is important to check for class imbalance in the target variable. If there is a significant class imbalance, where the number of churned customers is much smaller than the number of non-churned customers, it may affect the model's performance and require special consideration during the analysis.

Exploratory Data Analysis (EDA): Before applying machine learning models, performing EDA on the dataset can provide insights into the relationships between different features and the target variable. Visualizations, such as bar plots, histograms, and correlation matrices, can help identify patterns, trends, and potential correlations.

Feature Engineering: Depending on the specific analysis goals, feature engineering techniques, such as one-hot encoding, label encoding, or creating new derived features, may be applied to enhance the predictive power of the models.

Model Performance: The accuracy, precision, recall, and F1-score values reported for each model give an indication of their performance in predicting customer churn. Higher values indicate better model performance, but it's important to consider the specific goals and requirements of the analysis.

Model Comparison: By comparing the performance of different models (e.g., Random Forest, Logistic Regression, XGBoost, AdaBoost, LightGBM), you can identify the models that provide the best predictive power for customer churn in this dataset. Each model has its own strengths and weaknesses, so it's important to consider multiple models and their respective performance metrics.

These are just some general observations based on the provided dataset information. For a more detailed analysis, it would be beneficial to explore the dataset further, perform EDA, and assess the significance of different features in predicting customer churn.

# Conclusion 
RandomForestClassifier achieved the best accuracy of 0.8041 with the following hyperparameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 200}.

LogisticRegression achieved a slightly higher accuracy of 0.8197 with the hyperparameters {'C': 0.1, 'penalty': 'l2'}.

SVC achieved an accuracy of 0.8176 with the hyperparameters {'C': 10, 'kernel': 'linear'}.

KNeighborsClassifier achieved an accuracy of 0.7757 with the hyperparameters {'n_neighbors': 7, 'weights': 'uniform'}.

DecisionTreeClassifier achieved an accuracy of 0.7715 with the hyperparameters {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 10}.

The Keras Neural Network (assuming it's a multi-layer perceptron) achieved an accuracy of 0.8013 with the hyperparameters {'optimizer': 'adam', 'neurons': 32, 'activation': 'relu'}.

XGBoost achieved an accuracy of 0.8077, precision of 0.6747, recall of 0.5282, and F1-score of 0.5925.

AdaBoost achieved a mean accuracy of 0.7991 with a standard deviation of 0.0094.

LightGBM achieved an accuracy of 0.8126, precision of 0.6823, recall of 0.5469, and F1-score of 0.6071.

Based on these results, LogisticRegression, SVC, and LightGBM models seem to perform better with relatively higher accuracy compared to other models. However, the performance of the models may vary depending on the specific dataset and problem domain. It is recommended to further evaluate the models on different metrics and perform additional testing to make a more informed decision.
