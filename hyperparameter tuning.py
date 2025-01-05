from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from functions import data_processing, scaling, train_test_sets, remove_correlated_variables, calculate_descriptors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def hyperparameters_randomsearch(filename):
    """
    Perform hyperparameter tuning for a RandomForestClassifier using RandomizedSearchCV.
    """
    n_estimators = [int(x) for x in np.linspace(start=500, stop=1500, num=15)]
    max_features = ['log2', 'sqrt', 'auto']
    max_depth = [int(x) for x in np.linspace(20, 400, num=12)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=6)]
    min_samples_leaf = [int(x) for x in np.linspace(start=1, stop=10, num=6)]
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

    training_data = data_processing(filename, True)
    train_descriptors = calculate_descriptors(training_data)
    final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
    data = remove_correlated_variables(final_df, 0.9)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_sets(data)
    X_train_scaled = scaling(X_train, scaler)
    X_test_scaled = scaling(X_test, scaler, False)

    rf_random.fit(X_train_scaled, y_train)

    best_params = rf_random.best_params_

    best_model = rf_random.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy with best model: {balanced_accuracy}")

    return best_params

#best_params_random = hyperparameters_randomsearch("train.csv")
#print(f"Best hyperparameters from random search: {best_params_random}")

def hyperparameters_gridsearch(filename):
    """
    Perform hyperparameter tuning for a RandomForestClassifier using GridSearchCV.
    """
    param_grid = {
        'n_estimators': [1000, 1200, 1400],
        'max_depth': [120, 160, 190, None],
        'max_features': ['sqrt', 'log2', 'auto'],
        'min_samples_split': [6, 8, 10],
        'min_samples_leaf': [2, 4, 6],
        'bootstrap': [True, False],
        'criterion' : ["gini", 'entropy'],
        'class_weight' : ['balanced', 'balanced_subsample']}

    training_data = data_processing(filename, True)
    train_descriptors = calculate_descriptors(training_data)
    final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
    data = remove_correlated_variables(final_df, 0.9)
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_sets(data)
    X_train_scaled = scaling(X_train, scaler)
    X_test_scaled = scaling(X_test, scaler, False)

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring='balanced_accuracy')

    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy of the best model: {balanced_accuracy}")
    
run = hyperparameters_gridsearch("train.csv")
