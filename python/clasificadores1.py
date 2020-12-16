# -*- coding: utf-8 -*-

import numpy
from pandas import read_csv
from matplotlib import pyplot
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Cargar dataset
# dataset = read_csv('./dataset_valence_2.5_7.5.csv', header=None)
# dataset = read_csv('./dataset_valence_3_7.csv', header=None)
# dataset = read_csv('./dataset_valence_4_6.csv', header=None)
# dataset = read_csv('./dataset_valence_5_5.csv', header=None)
# dataset = read_csv('./dataset_arousal_2.5_7.5.csv', header=None)
# dataset = read_csv('./dataset_arousal_3_7.csv', header=None)
# dataset = read_csv('./dataset_arousal_4_6.csv', header=None)
# dataset = read_csv('./dataset_arousal_5_5.csv', header=None)

dataset = read_csv('../caracteristicas/dataset1_m1_3_7.csv')

array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]

unique, counts = numpy.unique(Y, return_counts=True)
print("Muestras por clase")
print(dict(zip(unique, counts)))

# Dividir en sets de entrenamiento y test
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

num_folds = 10
scoring = 'accuracy' # Métrica a utilizar para la evaluación

##########################
# Evaluating algorithms
##########################

models = []
models.append(( 'KNN' ,  KNeighborsClassifier()))
models.append(( 'TREE' , DecisionTreeClassifier()))
models.append(( 'NB' , GaussianNB()))
models.append(( 'SVM' , SVC()))

results = []
names = []

print("")
print("COMPARACIÓN MODELOS SIN ESCALADO DE DATOS")
print("-----------------------------------------")
for name, model in models:
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Prueba grafik
DEPTHS = list(range(2,6))
depth_accuracy  = results[0]
depths_accuracy = results
depth_mean      = numpy.mean(depth_accuracy)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4), gridspec_kw={'width_ratios': [1, 2]})
fig.suptitle("ANALISIS MONTECARLO\n%d experimentos" % (10))
    # Histograma
ax1.hist(100*depth_accuracy, bins=20, ec='black')
props = dict(facecolor='white', alpha=10)
ax1.text(0.40, 0.94, "media: %.1f%%" % (100*depth_mean), transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
ax1.set(xlabel='Tasa acierto [%]', ylabel='Ocurrencias', title="Mejor: MAX_DEPTH=%d" % (1))
    # Boxplot simple
# ax2.boxplot(100*depth_accuracy)
    # Step
# ax2.step(range(0,10), depth_accuracy, label='pre (default)')
# ax2.plot(range(0,10), depth_accuracy, 'C0o', alpha=0.5)
# ax2.legend()
    # Boxplot todos
# ax2.adjust(left=0.4)
ax2.boxplot(depths_accuracy)
ax2.set(xlabel="MAX_DEPTH", ylabel="Tasa acierto [%]")
ax2.set_xticklabels(DEPTHS)
# ax2cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.subplots_adjust(wspace=0.44, top=0.84)

fig.savefig('aaaaa.png')

exit()


# boxplot algorithm comparison
# Comparamos los distintos métodos
fig = pyplot.figure()
fig.suptitle('Comparación algoritmos. Datos sin escalar.')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

###########################################
# Escalar datos a media 0 y varianza unidad
###########################################

scaler = StandardScaler().fit(X)
scaledX_train = scaler.transform(X_train)
scaledX_validation = scaler.transform(X_validation)
# scaledX_train = X_train
# scaledX_validation = X_validation

results = []
names = []

print("")
print("COMPARACIÓN MODELOS CON ESCALADO DE DATOS")
print("-----------------------------------------")
for name, model in models:
    
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, scaledX_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
# Comparamos los distintos métodos
fig = pyplot.figure()
fig.suptitle('Comparación algoritmos. Datos escalados.')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()


############################################
# Modificar parámetros de los clasificadores
############################################

# K Neighbors sin escalar datos
###############################
print("")
print("        KNN SIN ESCALADO DE DATOS        ")
print("-----------------------------------------")

# k = 7 valor por defecto
neighbors = [15,17,19,21,23,25,27,28,29,30,31,32,33,34,35]

param_grid = {'n_neighbors': neighbors}
model = KNeighborsClassifier()

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(X_train,Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]

for i in range(len(means)): 
    print("%f (%f) with: %r" % (means[i], stds[i], params[i]))


# K Neighbors datos escalados
#############################
print("")
print("        KNN CON ESCALADO DE DATOS        ")
print("-----------------------------------------")

# k = 7 valor por defecto
neighbors = [15,17,19,21,23,25,27,28,29,30,31,32,33,34,35]

param_grid = {'n_neighbors': neighbors}
model = KNeighborsClassifier()

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold) 
grid_result = grid.fit(scaledX_train,Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]

for i in range(len(means)): 
    print("%f (%f) with: %r" % (means[i], stds[i], params[i]))
    
# SVC
######
print("")
print("        SVM CON ESCALADO DE DATOS        ")
print("-----------------------------------------")

# C = 1 y kernel Radial Basis (RBF) son los valores por defecto
    
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ]

param_grid = {'C' : c_values, 'kernel': kernel_values} 
model = SVC()

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(scaledX_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_[ 'mean_test_score' ]
stds = grid_result.cv_results_[ 'std_test_score' ]
params = grid_result.cv_results_[ 'params' ]

for i in range(len(means)): 
    print("%f (%f) with: %r" % (means[i], stds[i], params[i]))
    

############################
# Finalize training 
############################
print("")
# Probar modelos

# SVC    
# C = 1.3
# kernel = 'linear'
# print("")
# print("SVM sin escalar datos: C=" + str(C) + ", kernel=" + kernel)
# print("--------------------------------------------")
# model = SVC(C=C, kernel=kernel)
# model.fit(X_train, Y_train)
# result = model.score(X_validation, Y_validation)

# print("Accuracy: " + str(result))

# predicted = model.predict(X_validation)

# matrix = confusion_matrix(Y_validation, predicted)

# print("Confusion matrix: ")
# print(matrix)

# SVC datos escalados
C = 0.7
kernel = 'linear'
print("")
print("SVM datos escalados: C=" + str(C) + ", kernel=" + kernel)
print("--------------------------------------------")
model = SVC(C=C, kernel=kernel)
model.fit(scaledX_train, Y_train)
result = model.score(scaledX_validation, Y_validation)

print("Accuracy: " + str(result))

predicted = model.predict(scaledX_validation)

matrix = confusion_matrix(Y_validation, predicted)

print("Confusion matrix: ")
print(matrix)

# K Neighbors sin escalar datos
K = 17
print("")
print("KNN sin escalar datos: K=" + str(K))
print("----------------------------")
model = KNeighborsClassifier(n_neighbors=K)
model.fit(X_train, Y_train)
result = model.score(X_validation, Y_validation)

print("Accuracy: " + str(result))

predicted = model.predict(X_validation)

matrix = confusion_matrix(Y_validation, predicted)

print("Confusion matrix: ")
print(matrix)

# K Neighbors datos escalados
K = 17
print("")
print("KNN datos escalados: K=" + str(K))
print("----------------------------")
model = KNeighborsClassifier(n_neighbors=K)
model.fit(scaledX_train, Y_train)
result = model.score(scaledX_validation, Y_validation)

print("Accuracy: " + str(result))

predicted = model.predict(scaledX_validation)

matrix = confusion_matrix(Y_validation, predicted)

print("Confusion matrix: ")
print(matrix)