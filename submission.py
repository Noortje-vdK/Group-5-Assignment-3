from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

data_file = "train.csv"
data = pd.read_csv(data_file)

df = data.copy()
df = df.drop_duplicates()
df = df.dropna()
df = df.drop(df[(df['target_feature'] != 0) & (df['target_feature'] != 1)].index)

def getMolDescriptors(mol, missingVal=None):
    descriptors = {}
    for name, function in Descriptors.descList:
        try:
            value = function(mol)
        except Exception as e:
            print(f"Error calculating {name}: {e}")
            value = missingVal
        descriptors[name] = value
    return descriptors

def calculate_descriptors(df):
    all_descriptors = []
    for index, row in df.iterrows():
        smile_molc = row["SMILES_canonical"]
        mol = Chem.MolFromSmiles(smile_molc)
        if mol is None:
            print(f"Incorrect SMILES: {smile_molc}")
            continue
        molc_descr = getMolDescriptors(mol)
        all_descriptors.append(molc_descr)
    descriptors_df = pd.DataFrame(all_descriptors)
    return descriptors_df

descriptors_df = calculate_descriptors(df)
result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1)

def train_test_sets(X):
    input = X.drop(columns=["SMILES_canonical", "target_feature"], axis=1)
    output = X["target_feature"]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_sets(result_df)

random_forest = RandomForestClassifier(bootstrap= False, max_depth= 20, max_features= 'sqrt', min_samples_leaf= 4, min_samples_split= 2, n_estimators= 50, random_state=42, class_weight = "balanced")
random_forest.fit(X_train, y_train)

new_data = pd.read_csv('test.csv')
new_data = new_data.dropna()
new_data = new_data.drop_duplicates()
new_data_features = new_data.drop(columns=['Unique_ID'])

new_data_descriptors_df = calculate_descriptors(new_data_features)

if not X_train.columns.equals(new_data_descriptors_df.columns):
    print("Warning: Columns do not match! Re-aligning columns...")
    new_data_descriptors_df = new_data_descriptors_df.reindex(columns=X_train.columns, fill_value=0)

new_data_descriptors_df = new_data_descriptors_df.fillna(0)

# Make predictions using the trained model
predictions = random_forest.predict(new_data_descriptors_df)
#print("Predictions:", predictions)
#print("Prediction distribution:", pd.Series(predictions).value_counts())

# Add predictions to the original data
new_data['target_feature'] = predictions
new_data['target_feature'] = new_data['target_feature'].astype(str)
#print(new_data.dtypes)
#print(new_data[['Unique_ID', 'target_feature']].head(20))  # Print the first 20 rows for inspection

new_data[['Unique_ID', 'target_feature']].to_csv('predictions_output8.csv', index=False, quoting=1)

print("Predictions saved to 'predictions_output.csv'")