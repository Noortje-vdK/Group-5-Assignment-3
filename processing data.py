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

df.head()


