from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from functions import data_processing, train_test_sets, calculate_descriptors, getMolDescriptors
from sklearn.metrics import balanced_accuracy_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 1, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 25, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#pprint(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier(criterion='gini', random_state=42, class_weight = "balanced")
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model

training_data = data_processing("train.csv",True)
train_descriptors = calculate_descriptors(training_data)
final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
X_train, X_test, y_train, y_test = train_test_sets(final_df)

base_model = RandomForestClassifier(n_estimators=5, criterion='gini', max_depth=6, min_samples_split=2, min_samples_leaf=1, max_features = 'sqrt', random_state=42, class_weight = "balanced")
base_model.fit(X_train, y_train)
y_pred=base_model.predict(X_test)
balanced_accuracy_base = balanced_accuracy_score(y_test, y_pred)
print(f"the balanced accuracy of base model is {balanced_accuracy_base}.")

rf_random.fit(X_train, y_train)
rf_random.best_params_
print(rf_random.best_params_)

best_random = rf_random.best_estimator_

new_model = RandomForestClassifier(n_estimators=100, min_samples_split = 2, min_samples_leaf = 2, max_features = "sqrt", max_depth = 25, bootstrap = False, class_weight= "balanced", random_state= 42)
new_model.fit(X_train, y_train)
y_pred2 = new_model.predict(X_test)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred2)
print(f"the balanced accuracy of new model is {balanced_accuracy}.")


