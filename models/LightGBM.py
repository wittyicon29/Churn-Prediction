import lightgbm as lgb

# Convert the preprocessed data to LightGBM Dataset format
train_data = lgb.Dataset(X_train, label=y_train)

# Set the hyperparameters for LightGBM
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the LightGBM model
lgb_model = lgb.train(params, train_data, num_boost_round=100)

# Predict on the test set
y_pred_lgb = lgb_model.predict(X_test)
y_pred_lgb_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_lgb]

# Evaluate the LightGBM model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lgb_acc = accuracy_score(y_test, y_pred_lgb_binary)
lgb_precision = precision_score(y_test, y_pred_lgb_binary)
lgb_recall = recall_score(y_test, y_pred_lgb_binary)
lgb_f1 = f1_score(y_test, y_pred_lgb_binary)

print('LightGBM Accuracy:', lgb_acc)
print('LightGBM Precision:', lgb_precision)
print('LightGBM Recall:', lgb_recall)
print('LightGBM F1-score:', lgb_f1)
