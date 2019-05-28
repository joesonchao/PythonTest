import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
color = ['red', 'green']
marker = ['o', 'd']
print(X[0])
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=color[type], marker=marker[type])
    index += 1
plt.show()

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
export_graphviz(classifier1, out_file='graph\\demo19.dot', filled=True, rounded=True,
                special_characters=True)
from subprocess import check_call


#check_call(['dot', '-Tpng', '.\\graph\\demo19.dot', '-o', '.\\graph\\Output1.png'])
check_call(['dot', '-Tsvg', '.\\graph\\demo19.dot', '-o', '.\\graph\\Output1.svg'])

# Mac版本的 graphviz安裝套件怪怪的，無法跑出圖片
