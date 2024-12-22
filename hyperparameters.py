from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from functions import data_processing, train_test_sets, calculate_descriptors

def find_best_hyperparameters(data_path):
    """
    Perform hyperparameter tuning for a RandomForestClassifier using RandomizedSearchCV.

    Args:
        data_path (str): Path to the training data file.

    Returns:
        dict: Best hyperparameters found during tuning.
    """
    # Hyperparameter grid
    n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 50, num=10)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    # Base model for RandomizedSearchCV
    rf = RandomForestClassifier(criterion='gini', random_state=42, class_weight="balanced")

    # RandomizedSearchCV setup
    rf_random = RandomizedSearchCV(
        estimator=rf,
        param_distributions=random_grid,
        n_iter=100,
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Process data
    training_data = data_processing(data_path, True)
    train_descriptors = calculate_descriptors(training_data)
    final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)
    X_train, X_test, y_train, y_test = train_test_sets(final_df)

    # Fit RandomizedSearchCV
    rf_random.fit(X_train, y_train)

    # Get best hyperparameters
    best_params = rf_random.best_params_

    # Evaluate performance with best parameters
    best_model = rf_random.best_estimator_
    y_pred = best_model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    print(f"Best hyperparameters: {best_params}")
    print(f"Balanced accuracy with best model: {balanced_accuracy}")

    return best_params


best_params = find_best_hyperparameters("train.csv")
print(f"Best hyperparameters: {best_params}")


