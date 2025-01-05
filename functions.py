from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def data_processing(filename, testing = True):
    """Reads in data from file, removes rows with missing data and duplicate rows, and if the data is training data, 
    it removes non-binary target values."""
    data = pd.read_csv(filename)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    if testing:
        data = data.drop(data[(data['target_feature'] != 0) & (data['target_feature'] != 1)].index) # remove row if target feature is not binary
    return data

def getMolDescriptors(mol, missingVal=None):
    """For given molecule, calculate all molecular descriptors and returns it in a dictionary."""
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
    """For molecules in SMILES format, convert it to a RDkit molecule and calculate all descriptors for each molcule in dataframe."""
    all_descriptors = []
    for index, row in df.iterrows():
        smile_molc = row["SMILES_canonical"]
        mol = Chem.MolFromSmiles(smile_molc) # get an RDkit molecule from the SMILES
        if mol is None:
            print(f"Incorrect SMILES: {smile_molc}")
            continue
        molc_descr = getMolDescriptors(mol) # calculate all descriptors
        all_descriptors.append(molc_descr)
    descriptors_df = pd.DataFrame(all_descriptors)
    return descriptors_df

def remove_correlated_variables(data, threshold):
    """From data that only has numerical or binary columns, remove columns with variables that are highly correlated: 
    that have a correlation value above the threshold."""
    corr_matrix = data.drop(columns=["target_feature"]).corr()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)) # look only at upper triangle without diagonal to prevent removing both variables if correlated
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)] 
    return data.drop(columns=to_drop)

def train_test_sets(X):
    """Splits dataframe X in training (0.8) and test (0.2) set."""
    input = X.drop(columns=["SMILES_canonical", "target_feature"], axis=1) # input is only the descriptors
    output = X["target_feature"]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def scaling(data, scaler, fit = True):
    """Scales the data with given scaler, if scaler needs to be fitted to the data, it does so, 
    if it is new data, it only scales it with the existing scaler."""
    if fit:
        return scaler.fit_transform(data)
    else:
        return scaler.transform(data)

def submission(model, newfile, scaler, testfile="test.csv"):
    """Creates a csv file that predicts the target feature for testfile"""
    new_data = data_processing(testfile, False)
    new_descriptors = calculate_descriptors(new_data)
    new_descriptors.fillna(0, inplace= True)
    new_descriptors_scaled = scaling(new_descriptors, scaler, False)
    new_descriptors_scaled = pd.DataFrame(new_descriptors_scaled, columns=new_descriptors.columns, index=new_descriptors.index) # keep feature names
    predictions = model.predict(new_descriptors_scaled)
    new_data['target_feature'] = predictions
    new_data['target_feature'] = new_data['target_feature'].astype(str) # prevent that zeroes get removed while writing the file
    new_data[['Unique_ID', 'target_feature']].to_csv(newfile, index=False, quoting=1) # quoting = 1 ensures that dataframa is copied entirely to csv file