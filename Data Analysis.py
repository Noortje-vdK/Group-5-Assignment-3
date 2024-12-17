from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

data_file = "train.csv"
data = pd.read_csv(data_file)

df = data.copy()  # copys data to not change the original data
df = df.drop_duplicates() 
df = df.dropna()
df = df.drop(df[(df['target_feature'] != 0) & (df['target_feature'] != 1)].index) # remove data where the output is nonbinary

def getMolDescriptors(mol, missingVal=None):
    """
    Calculates the full list of descriptors for a molecule.
    If a descriptor cannot be calculated, missingVal is used.
    """
    descriptors = {}  # Initialize dictionary to store descriptors
    for name, function in Descriptors.descList:  # Iterate through descriptor functions
        try:
            value = function(mol)  # Calculate the descriptor
        except:
            import traceback
            traceback.print_exc()  # print error message if an error occurs
            value = missingVal  # use missingval when there is an error
        descriptors[name] = value  # add the descriptor and its value to the dictionary
    return descriptors

all_descriptors = []  # list to store descriptor dictionaries for all molecules

for index, row in df.iterrows():
    smile_molc = row.iloc[0]  # get smiles from first column
    mol = Chem.MolFromSmiles(smile_molc) 

    if mol is None:  # if the rdkit molecule is not usable
        print(f"Incorrect SMILES: {smile_molc}")
        continue  # skip it

    molc_descr = getMolDescriptors(mol)  # make descriptor disctonary for each molecule
    all_descriptors.append(molc_descr)  # append it to the list

descriptors_df = pd.DataFrame(all_descriptors) # make that list into dataframe for easy handling

result_df = pd.concat([df.reset_index(drop=True), descriptors_df], axis=1) # concatenate it with original data

print(result_df.head())

def train_test_sets(X):
    input = X.drop("target_feature", axis=1)  # Features
    output = X["target_feature"]  # Target labels
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_sets(result_df)
