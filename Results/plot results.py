import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Initialize the models (already tuned)
models = [
    (rf_model, 'Random Forest'),
    (logreg_model, 'Logistic Regression'),
    (svm_model, 'Support Vector Machines'),
    (knn_model, 'K-Nearest Neighbors'),
    (dt_model, 'Decision Trees')
]

# Fit the models with training data
for model, _ in models:
    model.fit(X_train, y_train)

# Plot ROC curve and print accuracy for each model
plt.figure(figsize=(10, 8))

for model, model_name in models:
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = model.score(X_test, y_test)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f}, Acc = {accuracy:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
