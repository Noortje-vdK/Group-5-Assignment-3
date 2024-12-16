from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


file = pandas.read_csv("breast_cancer.csv") #Use the pandas module to extract the data and put it in a dictionary called file
"""
Random seed implementeren. Read.ME file en resultaten instellen.
"""
def data_extractor(file):
    """
    Function to convert the file to a dataset split in data points
    """
    attributes = []
    data = []
    for line in file:
        attributes.append(line) #Make a list of all names of the attributes
    for i in range(len(attributes)):
        data.append(list(file[str(attributes[i])])) #Add all values to the list data 


    all_samples = []
    i = 0
    while i < len(data[0]): #Loop over all attributes
        sample = []
        for j in range(len(attributes)): #Loop over all values in an attributes
            sample.append(data[j][i]) #Add the value to its original sample
        all_samples.append(sample) #Add all samples togheter in a dataset
        i += 1

    random.shuffle(all_samples) #Shuffle the samples around

    return all_samples

data = data_extractor(file)
train, test = train_test_split(data)
print(train)
# random_forest = RandomForestClassifier()
# random_forest.fit(X_train, Y_train):
# Y_test = random_forest.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
