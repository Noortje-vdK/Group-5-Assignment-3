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

X_train = final_df.drop(columns=["SMILES_canonical", "target_feature"], axis=1)
y_train = final_df["target_feature"]
#X_train, X_test, y_train, y_test = train_test_sets(final_df)
random_forest = RandomForestClassifier(n_estimators= 1000, min_samples_split= 8, min_samples_leaf= 4, max_features= 'auto', max_depth= 15, criterion= 'entropy', class_weight= 'balanced_subsample', bootstrap= False)
random_forest.fit(X_train, y_train)

#y_pred = random_forest.predict(X_test)
#accuracy = accuracy_score(y_test, y_pred)
#balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy)
#print("Balanced Accuracy:", balanced_accuracy)
test = submission(random_forest, "predictions_output8.csv")

