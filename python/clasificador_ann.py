# -*- coding: utf-8 -*-
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
# Implementacion y ajuste de una red neuronal utilizando directamente Keras
from math import floor
from pandas import read_csv
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers  import Dense
from keras import optimizers
from sklearn.decomposition import PCA
from tensorflow import set_random_seed

import numpy
from matplotlib import pyplot as plt

import sys


if (len(sys.argv) != 7):
    print("Argumentos invalidos" + str(len(sys.argv)))
    exit(1)

DATASET_FILE = sys.argv[1]
seed         = int(sys.argv[2])
BATCH_SIZE   = int(sys.argv[3])
EPOCHS       = int(sys.argv[4])
resultad_tmp = sys.argv[5]
model_path   = sys.argv[6]

# seed = 7
numpy.random.seed(seed)
set_random_seed(seed)

# DIR_DATASET  = '../caracteristicas/'
# DIR_DATASET  = '../caracteristicas/multiple/'

# Cargar dataset

# DATASET_FILE = DIR_DATASET + 'dataset2_valence_2.5_7.5.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_valence_3_7.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_valence_4_6.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_valence_5_5.csv'

# DATASET_FILE = DIR_DATASET + 'dataset1_valence_deap_3_7.csv'
# DATASET_FILE = DIR_DATASET + 'dataset1_valence_todo_3_7.csv'

# DATASET_FILE = DIR_DATASET + 'dataset2_arousal_2.5_7.5.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_arousal_3_7.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_arousal_4_6.csv'
# DATASET_FILE = DIR_DATASET + 'dataset2_arousal_5_5.csv'

# DATASET_FILE = DIR_DATASET + 'dataset1_arousal_deap_3_7.csv'
# DATASET_FILE = DIR_DATASET + 'dataset1_arousal_todo_3_7.csv'

# NTRIALS_FILE   = DIR_DATASET + 'ntrials_valence_paciente_3_7.csv'

# Paciente con el que clasificar
PACIENTE = 1

dataset = read_csv(DATASET_FILE, header=None)
# ntrials = read_csv(NTRIALS_FILE, header=None)

# Numero de trials por paciente
# array  = ntrials.values
# inicio = numpy.sum(array[0:PACIENTE-1,0])
# fin    = numpy.sum(array[0:PACIENTE,0])

# print("inicio: " + str(inicio) + ", fin: " + str(fin))

# Dataset: Nos quedamos con los trials del paciente que vamos a clasificar. 
# X = Caracteristicas, Y = Etiquetas
array = dataset.values
# X = array[inicio:fin,0:-1]
# Y = array[inicio:fin,-1]
X = array[:,0:-1]
Y = array[:,-1]

# Dividir en sets de entrenamiento y test
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

####### PCA ########
####################
# n_comps = 10
# pca = PCA(n_components = n_comps, svd_solver = "full", whiten=False)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_validation = pca.transform(X_validation)
# X = pca.transform(X)
####################
####################


# Creacion de modelo
# ACTIVATION_1 = "linear"
# ACTIVATION_2 = "sigmoid"
# ACTIVATION_3 = "relu"
# EPOCHS       = 200
# BATCH_SIZE   = 60

# ACTIVATION_1 = "relu"
# ACTIVATION_2 = "relu"
# ACTIVATION_3 = "sigmoid"
# EPOCHS       = 456
# BATCH_SIZE   = 60

# ACTIVATION_1 = "relu"
# ACTIVATION_2 = "linear"
# ACTIVATION_3 = "sigmoid"
# EPOCHS       = 446 
# BATCH_SIZE   = 60

# dataset2_valence_3_7 486 12-20-1 76%
# ACTIVATION_1 = "relu"
# ACTIVATION_2 = "relu"
# ACTIVATION_3 = "sigmoid"
# LOSS         = "mean_squared_error"
# OPTIMIZER    = sgd
# EPOCHS       = 486
# BATCH_SIZE   = 50

ACTIVATION_1 = "relu"
ACTIVATION_2 = "relu"
ACTIVATION_4 = "sigmoid"
ACTIVATION_3 = "sigmoid"
LOSS         = "mean_squared_error"
OPTIMIZER    = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0005)
# OPTIMIZER    = optimizers.Adagrad(lr=0.0105, epsilon=0.01, decay=0.000)
# OPTIMIZER    = optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
# OPTIMIZER    = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
# OPTIMIZER    = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
# EPOCHS       = 50 # 349
# BATCH_SIZE   = 80

INPUT_DIM = X.shape[1]

# Opciones de optimizador
# sgd = optimizers.SGD(lr=0.001, momentum=0.8, decay=0.00005)

# Compilacion del modelo 
acc = 0
# while (acc < 0.75):
# Ajuste
model = Sequential()
model.add(Dense(12, input_dim=INPUT_DIM, kernel_initializer = "uniform", activation=ACTIVATION_1))
model.add(Dense(20, kernel_initializer = "uniform", activation=ACTIVATION_2))
# model.add(Dense(20, kernel_initializer = "uniform", activation=ACTIVATION_4))
model.add(Dense(1, kernel_initializer = "uniform", activation=ACTIVATION_3))
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=["accuracy"])
h = model.fit(X_train, Y_train, epochs=EPOCHS, validation_data=(X_validation, Y_validation), batch_size=BATCH_SIZE)

# Resultados con datos de validacion
scores = model.evaluate(X_validation, Y_validation)
acc    = scores[1]
print ("\n%s: %.2f%%" % (model.metrics_names[1], acc*100))

# Resultados con todos los datos
scores = model.evaluate(X, Y)
acc2   = scores[1]
print ("\n%s: %.2f%%" % (model.metrics_names[1], acc*100))

# Guardar acc y acc2 en fichero temporal
acc_save = str(acc*100)[0:6] + "," + str(acc2*100)[0:6]
result=open(resultad_tmp, "w")
result.write(acc_save)
result.close()
# Guardar modelo
model.save(model_path)
exit()

predicted = model.predict(X_validation)
predicted = 1*(predicted>0.5)
matrix = confusion_matrix(Y_validation, predicted)
print("Confusion matrix: ")
print(matrix)

plt.close("all")   # Cerramos figuras para mayor claridad

acc = h.history["acc"]
loss = h.history["loss"]
acc_val = h.history["val_acc"]
loss_val = h.history["val_loss"]

f = plt.figure(1)
f.suptitle("acc (r) y acc_val (b)")
ax = f.add_subplot(111)
ax.plot(acc, "r")
ax.plot(acc_val, "b")

ax.grid(b=True)

f1 = plt.figure(2)
f1.suptitle("loss (r) y loss_val (b)")
ax1 = f1.add_subplot(111)
ax1.plot(loss, "r")
ax1.plot(loss_val, "b")

ax1.grid(b=True)
plt.pause(0.001)
plt.show()