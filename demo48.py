import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset1 = np.loadtxt("data\\diabetes.csv", delimiter=",", skiprows=1)
print(type(dataset1))
print(dataset1.shape)

# inputList = dataset1[:, 0:8]
# resultList = dataset1[:, 8]
inputList = dataset1[:, 0:-1]
resultList = dataset1[:, -1]
print("input shape={}".format(inputList.shape))
print("result shape={}".format(resultList.shape))
print(np.unique(resultList))  # 0/1 only, classification


def create_default_model(optimizer='adam', init='uniform'):
    model = Sequential()
    model.add(Dense(64, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


from keras.wrappers.scikit_learn import KerasClassifier

model1 = KerasClassifier(build_fn=create_default_model, verbose=0)
optimizers = ['rmsprop', 'adam', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
#optimizers = ['rmsprop']
#inits = ['normal']
#epochs = [50]
#batches = [5]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model1, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
print("best :%f, using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f, (%f), with %r\n" % (mean, stdev, param))