from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def data_processing(filename, train = True):
    data = pd.read_csv(filename)
    data.drop_duplicates()
    data.dropna()
    if train:
        data.drop(data[(data['target_feature'] != 0) & (data['target_feature'] != 1)].index)
    else:
        data.drop(columns=["Unique_ID"])
    return data 

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

def train_test_sets(X):
    input = X.drop(columns=["SMILES_canonical", "target_feature"], axis=1)
    output = X["target_feature"]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



def submission(model, filename):
    new_data = data_processing("test.csv", False)
    new_descriptors = calculate_descriptors(new_data)
    new_descriptors.fillna(0)
    predictions = model.predict(new_descriptors)
    new_data['target_feature'] = predictions
    new_data['target_feature'] = new_data['target_feature'].astype(str)
    new_data[['Unique_ID', 'target_feature']].to_csv(filename, index=False, quoting=1)