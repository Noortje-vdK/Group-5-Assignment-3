from rdkit import Chem
import pandas as pd
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split

#read and save data from cvs file
data_file = "train.csv"
data = pd.read_csv(data_file)

#preprocessing data
df = data #taking a copy of the original data
df = df.drop_duplicates() #drops the second occurrence of duplicated rows
df = df.dropna() #drops rows with missing values
df = df.drop(df[(df['target_feature'] != 0) & (df['target_feature'] != 1)].index) #drops rows that have a non-binary value for the column 'target_feature'

#calculating descriptors
def getMolDescriptors(mol, missingVal=None):
    """calculates the full list of descriptors for a molecule
    if a descriptor can not be calculated missingVal is used"""
    res = {} #initializing dictionary for the name of each decriptor and its value for the molecule
    for name, func in Descriptors.descList: #iterating through descriptors
        #checking for erros
        try:
            value = func(mol) #calculates the descriptor for the molecule
        except:
            import traceback
            traceback.print_exc() #prints error message if an error occurs
            value = missingVal #sets the value to missingVal if an error occurs
        
        res[name] = value #adding the descriptor and its value to the dictionary
    return res

for row in df.values:
    smile_molc = row[0] #saving the molecule in smile form in a variable
    mol = Chem.MolFromSmiles(smile_molc)
    molc_descr = getMolDescriptors(mol) #calculating the descriptors for the molecule
    print(molc_descr)
    break #break om beter te kunnen kijken waar het fout gaat

def train_test_sets(X):
    input = X.drop("target_feature", axis=1)
    output = X["target_feature"]
    X_train, X_test, y_train, y_test = train_test_split(input, output, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_sets(df)
print(X_train)


