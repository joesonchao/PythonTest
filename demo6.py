import  matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regressionData1 = datasets.make_regression(10,5)
print(type(regressionData1))
print(regressionData1[0].shap, regressionData1[1].shape)
