from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
from functions import data_processing, train_test_sets, calculate_descriptors

# Grid of hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [10, 20, 30, 50, None],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Prepare data
training_data = data_processing("train.csv", True)
train_descriptors = calculate_descriptors(training_data)
final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
X_train, X_test, y_train, y_test = train_test_sets(final_df)

# Base model
rf = RandomForestClassifier(criterion='gini', random_state=42, class_weight='balanced')

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='balanced_accuracy'
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Get best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

print(f"Best hyperparameters: {best_params}")
print(f"Balanced accuracy of the best model: {balanced_accuracy}")
 
#Best hyperparameters: {'bootstrap': False, 'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_estimators': 50}
#Balanced accuracy of the best model: 0.941155132556195