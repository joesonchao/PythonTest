from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
data = iris.data
target = iris.target

classifier1 = KNeighborsClassifier(n_neighbors=3)
score = model_selection.cross_val_score(classifier1, data, target, cv=5)
print(score)