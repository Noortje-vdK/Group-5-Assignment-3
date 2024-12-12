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

#dit moet nog even gecheckt worden
for row in df[df.columns[-1]]: #iterates though the rows
    if row != 1 and row != 0:
        df.drop(row, axis=0) #drops row if the binary column contains a non-binary value

df.head()

