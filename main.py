from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from functions import data_processing, getMolDescriptors, calculate_descriptors, train_test_sets, submission

training_data = data_processing("train.csv")
train_descriptors = calculate_descriptors(training_data)
final_df = pd.concat([training_data.reset_index(drop=True), train_descriptors], axis=1)

def testing_accuracy(dataframe):
    """Trains a random forest model with 80% of the training data to estimate its performance 
    by returning the accuracy and balanced accuracy."""
    X_train, X_test, y_train, y_test = train_test_sets(dataframe)
    random_forest = RandomForestClassifier(bootstrap= False, class_weight= 'balanced', criterion= 'gini', max_depth= 14, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 7, n_estimators= 1200)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Balanced Accuracy:", balanced_accuracy)

#testrun = testing_accuracy(final_df)

def create_submission_randomforest(dataframe, filename):
    """Trains a random forest model with all training data to create a file to use for submission in Kaggle."""
    X_train = final_df.drop(columns=["SMILES_canonical", "target_feature"], axis=1)
    y_train = final_df["target_feature"]
    random_forest = RandomForestClassifier(bootstrap= False, class_weight= 'balanced', criterion= 'gini', max_depth= 14, max_features= 'sqrt', min_samples_leaf= 5, min_samples_split= 7, n_estimators= 1200)
    random_forest.fit(X_train, y_train)
    submission(random_forest, filename)

run = create_submission_randomforest(final_df, "predictions_output10.csv")




