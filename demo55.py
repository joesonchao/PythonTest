import keras.utils as utils

orig = 3
NUM_DIGITS = 20
print("before conversion:", orig)
converted = utils.to_categorical(orig, NUM_DIGITS)
print("after convert:", converted)
TYPE = 10
converted2 = utils.to_categorical(orig, TYPE)
print("after convert:", converted2)
orig2 = 0
converted3 = utils.to_categorical(orig2, TYPE)

print("[2]after convert:", converted3)