import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn import tree

iris = datasets.load_iris()
data = iris.data
target = iris.target

tree1 = tree.DecisionTreeClassifier()
score = model_selection.cross_val_score(tree1, data, target, cv=5)
print(score)