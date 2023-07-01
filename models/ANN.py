import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

def create_model(optimizer='adam', activation='relu', neurons=16):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

param_dist = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'sigmoid'],
    'neurons': np.arange(32, 64, 128)
}

from keras.wrappers.scikit_learn import KerasClassifier

keras_model = KerasClassifier(build_fn=create_model, verbose=0)

random_search = RandomizedSearchCV(keras_model, param_dist, cv=5, scoring='accuracy', n_iter=10)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Keras Neural Network - Best Accuracy: {accuracy:.4f}, Best Parameters: {random_search.best_params_}")
