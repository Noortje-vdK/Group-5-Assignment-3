from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from processing data import X_train, X_test, Y_train, Y_test

random_forest = RandomForestClassifier(n_estimators=5, criterion='gini', max_depth=6, min_samples_split=2, min_samples_leaf=1, max_features = 'sqrt')
random_forest.fit(X_train, Y_train)
y_pred = random_forest.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)