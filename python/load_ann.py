from keras.models import load_model
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import sys

if (len(sys.argv) != 3):
    print("Argumentos invalidos" + str(len(sys.argv)))
    exit(1)
model_file = sys.argv[1]
seed       = int(sys.argv[2])

DIR_DATASET  = '../caracteristicas/'
DATASET_FILE = DIR_DATASET + 'dataset2_valence_3_7.csv'
dataset = read_csv(DATASET_FILE, header=None)

# Cargar modelo
model = load_model(model_file)

# Cargar y dividir set de datos
array = dataset.values
X = array[:,0:-1]
Y = array[:,-1]
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Resultados con datos de validacion
scores = model.evaluate(X_validation, Y_validation)
acc    = scores[1]
print ("\n%s: %.2f%%" % (model.metrics_names[1], acc*100))

predicted = model.predict(X_validation)
predicted = 1*(predicted>0.5)
matrix = confusion_matrix(Y_validation, predicted)
print("Confusion matrix: ")
print(matrix)

# Resultados con todos los datos
scores = model.evaluate(X, Y)
acc    = scores[1]
print ("\n%s: %.2f%%" % (model.metrics_names[1], acc*100))

predicted = model.predict(X)
predicted = 1*(predicted>0.5)
matrix = confusion_matrix(Y, predicted)
print("Confusion matrix: ")
print(matrix)