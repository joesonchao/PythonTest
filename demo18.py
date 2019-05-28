from sklearn import tree

x = [[0, 0], [1, 1]]
y = [0, 1]
classifier = tree.DecisionTreeClassifier()
classifier.fit(x, y)

print((classifier.predict([[2, 2], [2, -2], [-2, 2], [-2, -2]])))
