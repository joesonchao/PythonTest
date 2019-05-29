import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

df1 = pd.read_csv('data/sonar.all-data', header=None, prefix='X')
print(df1.shape)

data, labels = df1.iloc[:, :-1], df1.iloc[:, -1]
print(data.shape)
print(labels.shape)
# 更改最後一欄名稱
df1.rename(columns={'X60': 'Label'}, inplace=True)
#
classifier = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
classifier.fit(X_train, y_train)
y_predict = classifier.predict((X_test))
print("score=", classifier.score(X_test, y_test))
result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(classifier, data, labels, cv=5, groups=labels)
print(scores)
