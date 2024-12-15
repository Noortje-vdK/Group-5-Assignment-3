from rdkit import Chem
import pandas as pd

#read and save data from cvs file
data_file = "C:/Users/20234472/OneDrive - TU Eindhoven/Advanced programming & biomedical data analysis/Assignment3/train.csv"
data = pd.read_csv(data_file)

data.head(1) #print first 5 rows to check if the data is loaded correctly (does only work in jupiter notebook)

#preprocessing data
df = data #taking a copy of the origional data

df = df.drop_duplicates() #drops the second occurance of duplicated rows
df = df.dropna() #drows rows with missing values
df = df.drop(df[(df['target_feature'] != 0) & (df['target_feature'] != 1)].index) #drops rows that have a non-binary value for the column 'target_feature'

df.head() #print first 5 rows to check if changes were made correctly

#calculating descriptors
def getMolDescriptors(mol, missingVal=None):
    """calculates the full list of descriptors for a molecule
    if a descriptor can not be calculated missingVal is used"""
    res = {} #initializing dictionary for the name of each decriptor and its value for the molecule
    for name, func in Descriptors._descList: #iterating through descriptors
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
    molc_descr = getMolDescriptors(row) #calculating the descriptors for the molecule
    break #break om beter te kunnen kijken waar het fout gaat




