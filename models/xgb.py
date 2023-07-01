from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Create the XGBoost classifier
xgb_model = xgb.XGBClassifier()

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print('Best Hyperparameters:', best_params)

# Train the XGBoost model with the best hyperparameters
xgb_model_best = xgb.XGBClassifier(**best_params)
xgb_model_best.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model_best.predict(X_test)

# Evaluate the XGBoost model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

xgb_acc = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

print('XGBoost Accuracy:', xgb_acc)
print('XGBoost Precision:', xgb_precision)
print('XGBoost Recall:', xgb_recall)
print('XGBoost F1-score:', xgb_f1)
