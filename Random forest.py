import pandas
import numpy as np
import random
random.seed(10)

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



class Node():
    """
    The class node has a couple of characteristics that specify which path a datapoint should
    take down this node
    """
    def __init__(self,index=None, threshold=None, left=None, right=None, info_gain=None, value= None):

        """constructor"""

        self.index = index #Determines the index of the feature this Node uses to split data
        self.threshold = threshold #Value on which the data is split
        self.left = left #
        self.right = right 
        self.info_gain = info_gain #Stores the information gained on this Node
        self.value = value #In case of a leave, the value of the Node gives the predicted target value

class Decision_Tree():
    def __init__(self, min_samples_split= 20 , max_depth = 3):
        ''' constructor '''
        

        #Stopping conditions
        self.min_samples_split = min_samples_split #The minimum amount of samples a Node can have to be split farther
        self.max_depth = max_depth #The maximum amount of splits from the root a Node can have
        
    
    def train_test_split(self, data):
         """Split the dataset 60/20/20 into a training, validation and test dataset"""

         train_data, val_data, test_data = [], [], [] #Store each dataset in a list
         i = 0 #Initialize the index

         #Loop over all the data and add a certain amount of data to each dataset
         while i <= 0.6*len(data):
             train_data.append(data[i])
             i += 1
         while i <=0.8*len(data):
             val_data.append(data[i])
             i += 1
         while i < len(data):
             test_data.append(data[i])
             i += 1

         return train_data, val_data, test_data
    

    def grow_tree(self, data, depth = 0):
        """Create new nodes for the tree and return a leaf when the limit conditions are met"""

        best_split = self.best_split(data)#Calls the function best_split to receive a dictionary with the caractheristics of the best split
        num_samples = len(data) #Gives the number of datapoints left in this part of the tree
        if num_samples >= self.min_samples_split and depth < self.max_depth: #Checks the stopping conditions
            if best_split["info_gain"] > 0: #Checks whether the data can still be split, or is split perfectly
                left_tree = self.grow_tree(best_split["data_left"], depth + 1) #Recursion to the left part of the tree
                right_tree = self.grow_tree(best_split["data_right"], depth + 1) #Recursion to the right part of the tree
                return Node(best_split["index"], best_split["threshold"], left_tree, right_tree, best_split["info_gain"]) #Create a Node with the caractheristics of the best split
         
        #No better split so a leaf is created
        Y = [] #Create a list for all target values
        for value in data: #Loop over all datapoints to extract the target values
            Y.append(value[-1])
        leaf_value = max(Y, key=Y.count) #Set the value of the leaf to the Y that has the majority.
        return(Node(value= leaf_value))
    
    def best_split(self, data):
        """
        Search for which attribute and which threshold give the maximum information gain
        and thus the best split
        """

        best_split = {} #Create a dictionary for the charasteristics of the best split
        max_info_gain = -1 #Initialize the maximum information gain to lower than zero
    
        for i in range(len(data[0])-1):#Loop over all attributes, except the target value at the end
            possible_thresholds = set() #Make a set to store unique possible thresholds
            for j in range(len(data)): 
                possible_thresholds.add((data[j][i])) #Add all unique points to the possible thresholds
        
            for threshold in possible_thresholds: #Loop over all thresholds
                data_left, data_right, y, y_left, y_right = self.split(data, i, threshold) #split the data to check each threshold
                info_gain = self.get_info_gain(y, y_left, y_right) #Check the information gain using the get_info_gain function

                if info_gain > max_info_gain: #Checks whether the information gain is higher than the maximum 
                    best_split["index"] = i #The index of this attribute
                    best_split["threshold"] = threshold #The threshold
                    best_split["data_left"] = data_left #The data points where the point on the index i is lower or equal to the threshold
                    best_split["data_right"] = data_right #The data points where the point on the index i is higher than the threshold
                    best_split["info_gain"] = info_gain #The inforation gained
                    max_info_gain = info_gain
                
       
        #Return the dictionary containing all information of the best split
        return best_split

    def split(self, data, index, threshold):
        """Split the data into a left and right path using the threshold"""
        data_left, data_right, y, y_left, y_right = [], [], [], [], [] #Lists to split the data into 
        #Loop over all data_points
        for point in data: 
            if point[index] <= threshold: #Check whether the point on the given index is on or below the threshold
                data_left.append(point) 
                y_left.append(point[-1]) #Add the last value of the point to the target value list
            else:
                data_right.append(point)
                y_right.append(point[-1]) 
            y.append(point[-1]) #Add all target values to a list
        return data_left, data_right, y, y_left, y_right

    def get_info_gain(self, parent, child_left, child_right):
        """Calculate the information gain. A higher information gain gives a better splitpoint"""

        weight_l, weight_r = len(child_left)/len(parent), len(child_right)/len(parent) #Calculate the weight, the part of the data that is put left and right
        info_gain = self.gini(parent) - weight_l * self.gini(child_left) - weight_r * self.gini(child_right) #Calculate the information gain using the gini index
        return info_gain

    def gini(self, Y):
        """
        Calculate the gini_index, a measure of statistical inequality.
        The gini_index is used to calculate the information_gain
        """
        unique_y = list(np.unique(Y)) #Make a list with all possible values of y, in a binary case only [0,1]
        gini = 0
        for value in unique_y:
            probability = Y.count(value)/len(Y) #Calculate the probability that a value of y is equal to the selected target value
            gini += (probability)**2 #Formula for the gini index
        return 1 - gini #Return the information gain

    # def print_tree(self, tree=None, indent=" "): #Visualze the tree
    #     ''' Print the tree '''

    #     if tree.value is not None: #Check whether we have reached a leaf
    #         print(tree.value) 

    #     #Print the split nodes
    #     else:
    #         print("The decision tree:") 
    #         print("X_"+str(tree.index), "<=", tree.threshold, tree.info_gain)
    #         print("%sleft:" % (indent), end="")
    #         self.print_tree(tree.left, indent + indent)
    #         print("%sright:" % (indent), end="")
    #         self.print_tree(tree.right, indent + indent)
    
    def predict(self, test_data, root):
        """Predict the target value using the validation or test dataset"""
        X = []
        for value in test_data: #Remove the target values from the test or validation data
            X.append(value[:-1])
        
        predictions = []
        for x in X: #Loop over all datapoints
            predictions.append(self.make_prediction(x,root)) #Add the predictions to a list using the make_predition function
       
        return predictions
    
    def make_prediction(self, x, tree):
        '''Predict the target of a single sample '''
        
        if tree.value!=None: #Check whether we have reached a leaf nod
            
            return tree.value
           
        feature_val = x[tree.index] #Give the value of the datapoint at the given index
        if feature_val<=tree.threshold: #Check whether it is above or below the threshold
            return self.make_prediction(x, tree.left) #Recursion to go down the left path of the node
        else:
            return self.make_prediction(x, tree.right) #Recursion to go down the right path of the node

i = 0
all_predictions = []
nr_trees = 1
while i < nr_trees:
    Tree = Decision_Tree()#Make a new tree called Tree
    #train_data, val_data, test_data = Tree.train_test_split(data) #Split the data set
    root = Tree.grow_tree(data) #Make a root of the tree using 
    #Tree.print_tree(root) #Print the tree
    
    Y_pred = Tree.predict(data, root) #Make a list of all predicted y for the test data
    all_predictions.append(Y_pred)
    i += 1

predictions = []
print([sublist[0] for sublist in all_predictions])
for i in range(len(all_predictions[0])):

    
    predictions.append(sublist[i] for sublist in all_predictions)
    #print(predictions)


# correct = 0 #Initialize amount of correct predictions
# for i in range(len(Y_pred)): #Loopover all predicted targets
#     if Y_pred[i] == test_data[i][-1]: #Check if the prediction is the same as the validation set
#         correct += 1

# accuracy = correct/len(Y_pred) #Calculates the accuracy of the predictions
# print(f"The predictions of Y were {accuracy*100} percent accurate.")