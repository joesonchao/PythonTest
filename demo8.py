import numpy as np
from sklearn import  linear_model, datasets

diabestes = datasets.load_diabetes()
print(type(diabestes))
print(diabestes.data.shape)
print(diabestes.target.shape)
