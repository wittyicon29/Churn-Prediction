from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# Create an AdaBoost classifier
adaboost = AdaBoostClassifier()

# Perform cross-validation with AdaBoost
adaboost_scores = cross_val_score(adaboost, X_train, y_train, cv=5)

# Compute the mean accuracy and standard deviation
adaboost_mean_accuracy = adaboost_scores.mean()
adaboost_std_accuracy = adaboost_scores.std()

# Print the results
print("AdaBoost:")
print("Mean Accuracy:", adaboost_mean_accuracy)
print("Standard Deviation:", adaboost_std_accuracy)
