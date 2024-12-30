from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from functions import data_processing, train_test_sets, calculate_descriptors

def hyperparameters_randomsearch(data_path):
    """
    Perform hyperparameter tuning for a RandomForestClassifier using RandomizedSearchCV.
    """
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
    max_features = ['log2', 'sqrt', 'auto']
    max_depth = [int(x) for x in np.linspace(5, 100, num=10)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=4)]
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=10, num=4)]
    bootstrap = [True, False]
    criterion = ["gini", 'entropy']
    class_weight = ['balanced', 'balanced_subsample']

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap,
        'criterion': criterion,
        'class_weight': class_weight}

    rf = RandomForestClassifier(random_state=42)

    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=50,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1)

    training_data = data_processing(data_path, True)
    train_descriptors = calculate_descriptors(training_data)
    final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
    X_train, X_test, y_train, y_test = train_test_sets(final_df)

    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_

    best_model = rf_random.best_estimator_
    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy with best model: {balanced_accuracy}")

    return best_params

#best_params_random = hyperparameters_randomsearch("train.csv")
#print(f"Best hyperparameters from random search: {best_params_random}")

def hyperparameters_gridsearch(training_data):
    """
    Perform hyperparameter tuning for a RandomForestClassifier using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [1000, 1200, 1400],
        'max_depth': [12, 14, 16, None],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [7, 8, 9],
        'min_samples_leaf': [3, 4, 5],
        'bootstrap': [True, False],
        'criterion' : ["gini", 'entropy'],
        'class_weight' : ['balanced', 'balanced_subsample']}

    training_data = data_processing(training_data, True)
    train_descriptors = calculate_descriptors(training_data)
    final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
    X_train, X_test, y_train, y_test = train_test_sets(final_df)

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='balanced_accuracy')

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy of the best model: {balanced_accuracy}")
    
run = hyperparameters_gridsearch("train.csv")
