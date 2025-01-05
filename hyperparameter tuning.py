from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from functions import data_processing, scaling, train_test_sets, remove_correlated_variables, calculate_descriptors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler()

def hyperparameters_randomsearch(filename):
    """Perform hyperparameter tuning for a RandomForestClassifier using RandomizedSearchCV."""
    n_estimators = [int(x) for x in np.linspace(start=750, stop=1750, num=20)]
    max_features = ['sqrt', "log2", "auto", None]
    max_depth = [int(x) for x in np.linspace(70, 200, num=15)]
    max_depth.append(None)
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=12, num=6)]
    min_samples_leaf = [int(x) for x in np.linspace(start=4, stop=8, num=4)]
    bootstrap = [False]
    criterion = ['entropy', "gini", "log_loss"]
    class_weight = ['balanced', "balanced_subsample"]

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
    X_train, X_test, y_train, y_test = train_test_sets(final_df)
    X_train_scaled = scaling(X_train, scaler)
    X_test_scaled = scaling(X_test, scaler, False)

    rf_random.fit(X_train_scaled, y_train)

    best_params = rf_random.best_params_

    best_model = rf_random.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy with best model: {balanced_accuracy}")

best_params_random = hyperparameters_randomsearch("train.csv")

def hyperparameters_gridsearch(filename):
    """Perform hyperparameter tuning for a RandomForestClassifier using GridSearchCV."""
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
    X_train, X_test, y_train, y_test = train_test_sets(final_df)
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
    
#run = hyperparameters_gridsearch("train.csv")
