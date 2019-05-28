import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets

iris = datasets.load_iris()
df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df1.shape)
df1['species'] = np.array([iris.target_names[i] for i in iris.target])
print(df1.shape)

seaborn.pairplot(df1, hue='species')
plt.show()
plt.cla()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df1[iris.feature_names], iris.target,
                                                    test_size=0.5, stratify=iris.target)
from sklearn.ensemble import RandomForestClassifier

# oob => Out of bag 袋外樣本
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf1.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

predicted = rf1.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print('OOB score:{:.3}'.format(rf1.oob_score_))
print('mean accuracy:{:.3}'.format(accuracy))
from sklearn.metrics import confusion_matrix

cm1 = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
seaborn.heatmap(cm1, annot=True)
plt.show()
