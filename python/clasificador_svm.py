import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
import sys
import os
import code

from simplejson import load as json_load

from matplotlib import pyplot as plt

config_file = sys.argv[1]
if not os.path.isfile(config_file):
    print(f"Fichero {config_file} no encontrado")
    exit(1)

with open(config_file, 'r') as f:
    config = json_load(f)

DATASET_FILES = config['DATASET_FILES']
KERNELS       = config['KERNELS']
C             = config['C']
SEED1         = config['SEED1']
SEED2         = config['SEED2']
OUTLIERS      = config['OUTLIERS_PERCENT']
O_STEP        = config['OUTLIERS_STEP']
PCA1          = config['PCA1']
PCA2          = config['PCA2']
PCA_STEP      = config['PCA_STEP']
SAVE_DIR_     = config['SAVE_DIR']
MULTIPLICIDAD = config['MULTIPLICIDAD']

if  SAVE_DIR_[-1] != '/':
    SAVE_DIR_ += "/"

SEEDS = list(range(SEED1, SEED2+1))

b_combs = {}

# Hacer los análisis en cada conjunto
for x in range(len(DATASET_FILES)):

    DATASET_FILE = DATASET_FILES[x]

    pca1         = PCA1[x]
    pca2         = PCA2[x]
    pca_step     = PCA_STEP[x]

    print("")
    print("                  " + DATASET_FILE)
    print("------------------------------------------------------------------------------------")

    dataset = read_csv(DATASET_FILE, header=None)

    array = dataset.values
    X = array[:,0:-1]
    Y = array[:,-1]

    SAVE_DIR = SAVE_DIR_ + DATASET_FILE.split('/')[-1].split('.')[0] + "/"

    dset = DATASET_FILE.split('/')[-1].split('.')[0].split('_')[0]

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    #######################################################################################################################################
    ### ANÁLISIS MONTECARLO PARA KERNEL y C
    #######################################################################################################################################

    # KERNELS y C son ya listas

    kernels_accuracy = {}
    kernels_mean_acc = {}

    COMBS  = [(k, c) for k in KERNELS for c in C]
    XTICKS = [f"{k}\nC = {c}" for k in KERNELS for c in C]

    m = 0

    for kernel, c in COMBS:

        comb_accuracy = np.zeros(len(SEEDS))

        for i in range(len(SEEDS)):
            seed = SEEDS[i]

            # Dividir en sets de entrenamiento y test
            validation_size = 0.20
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

            # Escalar datos
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_validation = scaler.transform(X_validation)

            # Entrenar clasificador
            model = model = SVC(C=c, kernel=kernel)
            model.fit(X_train, Y_train)

            # Guardar accuracy
            acc  = model.score(X_validation, Y_validation)
            comb_accuracy[i] = acc

            # Borrar modelo
            del model

        # Guardar resultados para cada valor de C en cada kernel
        if kernel not in kernels_accuracy:
            kernel_accuracy = []
            kernel_mean_acc = np.zeros(len(C))
        else:
            kernel_accuracy = kernels_accuracy[kernel]
            kernel_mean_acc = kernels_mean_acc[kernel]
        
        comb_mean = comb_accuracy.mean()

        kernel_accuracy.append(comb_accuracy)
        kernel_mean_acc[m%len(C)] = comb_mean

        kernels_accuracy.update({kernel: kernel_accuracy})
        kernels_mean_acc.update({kernel: kernel_mean_acc})

        m += 1

#### Gráficas ####

    # De cada kernel
    kernels_b_acc = np.zeros(len(KERNELS))
    b_ckernel = {}
    for k in range(len(KERNELS)):

        kernel = KERNELS[k]
        
        ckernel_mean_acc = kernels_mean_acc[kernel]
        ckernel_accuracy = kernels_accuracy[kernel]
    
        b       = np.argmax(ckernel_mean_acc)
        b_c     = C[b]
        b_acc   = ckernel_mean_acc[b]
        b_accs  = ckernel_accuracy[b]
        print ("Kernel: %s -> mejor resultado: C=%f con %.2f acc" % (kernel, b_c, 100*b_acc))
        # Agregar al diccionario
        b_ckernel.update({kernel: b_c})
        kernels_b_acc[k] = b_acc
        # Gráfica resultado analisis montecarlo
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,4), gridspec_kw={'width_ratios': [4, 8]})
        fig.suptitle("ANÁLISIS MONTECARLO\n%d experimentos" % (len(SEEDS)))
            # Histograma mejor resultado
        ax1.hist(100*b_accs, bins=20, ec='black')
        props = dict(facecolor='white', alpha=10)
        ax1.text(0.61, 0.92, "media: %.1f%%" % (100*b_acc), transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
        ax1.set(xlabel='Tasa acierto [%]', ylabel='Ocurrencias', title="Mejor: C=%d" % (b_c))
            # Boxplot todos 
        ax2.boxplot(ckernel_accuracy)
        ax2.plot(range(1, len(C)+1), ckernel_mean_acc, linestyle='-', marker='.', alpha=0.7, color='#d62728')
        ax2.set(xlabel="C", ylabel="Tasa acierto")
        ax2.set_xticklabels(C)
            # Guardar figura
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.2, top=0.9)
        fig.savefig(SAVE_DIR+f'{kernel}_c.png')

    # Bar chart acc kernels
    b_c_s = []
    w = 0.7  # the width of the bars
    largo = max(5, (1.2*w*len(KERNELS)))
    fig, ax = plt.subplots(figsize=(largo,5))
    for k in range(len(KERNELS)):
        kernel = KERNELS[k]
        kernel_bc = b_ckernel[kernel]
        b_c_s.append(kernel_bc)
        kernel_b_acc = kernels_b_acc[k]
        rect = ax.bar((k+1)*w, 100*kernel_b_acc, w, label=kernel, edgecolor='black')
        x = (rect[0].get_x() + rect[0].get_width()/2.)
        y = 1.03*rect[0].get_height()
        ax.text(x, y, "%.1f%%" %(100*kernel_b_acc), ha='center', va='bottom', fontsize=10)
    ax.legend()
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True ,      # ticks along the bottom edge are on
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    r = np.arange(w, w*(1+len(KERNELS)), w)
    ax.set_xticks(r)
    ax.set_xticklabels(b_c_s)
    ax.set(xlabel='C', ylabel='Tasa acierto [%]', title="Tasa de acierto para los mejores valores de C")
    ax.set_ylim(ymax=100)
    fig.savefig(SAVE_DIR+'bar.png')

    # Mejores resultados de todos los kernels en cada dataset
    b_combs.update({dset: b_ckernel})

    # Mejor combinación: bk, bkc
    b = np.argmax(kernels_b_acc)
    bk  = KERNELS[b]
    bkc = b_ckernel[bk]
    print(f"Mejor combinacion: kernel {bk} con C={bkc} -> {100*kernels_b_acc[b]}%")

    bkernel_accs = kernels_accuracy[bk]
    bkernel_mean = kernels_mean_acc[bk]


    #######################################################################################################################################
    ### ANÁLISIS MONTECARLO PARA OUTLIERS
    #######################################################################################################################################

    O_PERCENTS = list(range(0, OUTLIERS+1, O_STEP))
    if  O_PERCENTS[-1] != OUTLIERS:
        O_PERCENTS.append(OUTLIERS)

    outliers_accuracy = []
    outliers_mean_acc = np.zeros(len(O_PERCENTS))

    for p in range(len(O_PERCENTS)):
        percent = O_PERCENTS[p]
        percent_accuracy = np.zeros(len(SEEDS))
        for i in range(len(SEEDS)):
            seed = SEEDS[i]

            # Dividir en sets de entrenamiento y test
            validation_size = 0.20
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

            # Escalar datos
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_validation = scaler.transform(X_validation)

            # Introducir outliers
            n_outliers = int(percent*len(Y_train)/100)
            outliers_i = np.random.choice(len(Y_train), n_outliers, replace=False)  
            Y_train[outliers_i] = 1 - Y_train[outliers_i]

            # Entrenar clasificador
            model = model = SVC(C=bkc, kernel=bk)
            model.fit(X_train, Y_train)

            # Guardar accuracy
            acc  = model.score(X_validation, Y_validation)
            percent_accuracy[i] = acc

            # Borrar modelo
            del model     

        # Guardar resultados para cada porcentaje de outliers diferente
        percent_mean = percent_accuracy.mean()
        outliers_accuracy.append(percent_accuracy)
        outliers_mean_acc[p] = percent_mean

    # Mejor resultado medio
    b         = np.argmax(outliers_mean_acc)
    b_percent = O_PERCENTS[b]
    b_acc     = outliers_mean_acc[b]
    b_accs    = outliers_accuracy[b]
    # Gráfica resultado analisis montecarlo
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,4), gridspec_kw={'width_ratios': [4, 7]})
    fig.suptitle("ANÁLISIS OUTLIERS: kernel=\'%s\', C=%.1f" % (bk, bkc))
        # Histograma mejor resultado
    ax1.hist(100*b_accs, bins=20, ec='black')
    props = dict(facecolor='white', alpha=10)
    ax1.text(0.61, 0.92, "media: %.1f%%" % (100*b_acc), transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax1.set(xlabel='Tasa acierto [%]', ylabel='Ocurrencias', title="Mejor: %d%% outliers" % (b_percent))
        # Boxplot todos 
    ax2.boxplot(outliers_accuracy)
    ax2.plot(range(1, len(O_PERCENTS)+1), outliers_mean_acc, linestyle='-', marker='.', alpha=0.7, color='#d62728')
    ax2.set(xlabel="%% outliers", ylabel="Tasa acierto")
    ax2.set_xticklabels(O_PERCENTS)
        # Guardar figura
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, top=0.9)
    fig.savefig(SAVE_DIR+'outliers.png')


    #######################################################################################################################################
    ### ANÁLISIS MONTECARLO PARA REDUCCIÓN DIMENSIONALIDAD CON PCA
    #######################################################################################################################################

    NCOMPS = list(range(pca1, pca2+1, pca_step))
    if  NCOMPS[-1] != pca2:
        NCOMPS.append(pca2)

    ncomps_accuracy = []
    ncomps_mean_acc = np.zeros(len(NCOMPS))

    for n in range(len(NCOMPS)):
        ncomp = NCOMPS[n]
        ncomp_accuracy = np.zeros(len(SEEDS))

        for i in range(len(SEEDS)):
            seed = SEEDS[i]

            # Dividir en sets de entrenamiento y test
            validation_size = 0.20
            X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

            # Reducción de dimensionalidad con PCA
            pca = PCA(n_components=ncomp, svd_solver = "full", whiten=False)
            pca.fit(X_train)
            X_train              = pca.transform(X_train)
            X_validation         = pca.transform(X_validation)

            # Escalar datos
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_validation = scaler.transform(X_validation)

            # Entrenar clasificador
            model = model = SVC(C=bkc, kernel=bk)
            model.fit(X_train, Y_train)

            # Guardar accuracy
            acc  = model.score(X_validation, Y_validation)
            ncomp_accuracy[i] = acc

            # Borrar modelo
            del model
            del pca
            
        # Guardar resultados para cada número de componentes diferente
        ncomp_mean = ncomp_accuracy.mean()
        ncomps_accuracy.append(ncomp_accuracy)
        ncomps_mean_acc[n] = ncomp_mean
            

    # Mejor resultado medio
    b         = np.argmax(ncomps_mean_acc)
    b_ncomp   = NCOMPS[b]
    b_acc     = ncomps_mean_acc[b]
    b_accs    = ncomps_accuracy[b]
    print ("Mejor resultado: %d componentes PCA con %.2f acc" % (b_ncomp, 100*b_acc))
    # Gráfica resultado análisis montecarlo
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,4), gridspec_kw={'width_ratios': [4, 7]})
    fig.suptitle("ANÁLISIS REDUCCIÓN DE DIMENSIONALIDAD CON PCA: kernel=\'%s\', C=%.1f" % (bk, bkc))
        # Histograma mejor resultado
    ax1.hist(100*b_accs, bins=20, ec='black')
    props = dict(facecolor='white', alpha=10)
    ax1.text(0.61, 0.92, "media: %.1f%%" % (100*b_acc), transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props)
    ax1.set(xlabel='Tasa acierto [%]', ylabel='Ocurrencias', title="Mejor: %d componentes" % (b_ncomp))
        # Boxplot todos 
    ax2.boxplot(ncomps_accuracy)
    ax2.plot(range(1, len(NCOMPS)+1), ncomps_mean_acc, linestyle='-', marker='.', alpha=0.7, color='#d62728')
    ax2.set(xlabel="nº componentes", ylabel="Tasa acierto")
    ax2.set_xticklabels(NCOMPS)
        # Guardar figura
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, top=0.9)
    fig.savefig(SAVE_DIR+'pca.png')

#######################################################################################################################################
### ANÁLISIS MULTIPLICIDAD
#######################################################################################################################################
for comparacion in MULTIPLICIDAD:

    mults    = comparacion['M']
    files    = comparacion['FILES']
    conjunto = comparacion['CONJUNTO']

    dset     = files[0].split('/')[-1].split('.')[0].split('_')[0]

    SAVE_DIR = SAVE_DIR_ + "multiplicidad/" + dset + "/"
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    mults_accuracy = []
    mults_mean_acc = np.zeros(len(mults))

    b_combs_d = b_combs[dset]

    mul_res = {}

    for m in range(len(mults)):
        mult    = mults[m]
        f       = files[m]
        data    = read_csv(f, header=None)
        array   = data.values
        X       = array[:,0:-1]
        Y       = array[:,-1]

        combs_res = {}
        
        m_entry   = f"MULTIPLICIDAD {mult}"

        for kernel in b_combs_d:
            c = b_combs_d[kernel]
            comb_accuracy = np.zeros(len(SEEDS))

            for i in range(len(SEEDS)):
                seed = SEEDS[i]

                # Dividir en sets de entrenamiento y test
                validation_size = 0.20
                X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

                # Escalar datos
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_validation = scaler.transform(X_validation)

                # Entrenar clasificador
                model = model = SVC(C=c, kernel=kernel)
                model.fit(X_train, Y_train)

                # Guardar accuracy
                acc  = model.score(X_validation, Y_validation)
                comb_accuracy[i] = acc

                # Borrar modelo
                del model

            comb_acc  = comb_accuracy.mean()
            comb_r    = {"C": c, "acc": comb_acc}
            combs_res.update({kernel: comb_r})
        mul_res.update({m_entry: combs_res})

    # Pintar resultados para cada set de datos
    # Bar chart acc kernels
    w   = 0.6  # the width of the bars
    sep = 1.5*w
    nmults   = len(mults)
    nkernels = len(KERNELS)
    largo = (w*nkernels+sep)*nmults
    fig, ax = plt.subplots(figsize=(largo,5))
    tickslabels = list(mul_res.keys())

        # Posicion xticks
    offset=w/2
    pos = []
    for x in range(nmults):
        posx = offset + nkernels*w/2+x*sep
        pos.append(posx)
        offset = offset + nkernels*w
    k = 0
    label    = True
    for multiplicidad in mul_res:
        for kernel in mul_res[multiplicidad]:
            c   = mul_res[multiplicidad][kernel]['C']
            acc = mul_res[multiplicidad][kernel]['acc']
            if label:
                rect = ax.bar((k+1)*w, 100*acc, w, label="%s C=%.1f" %(kernel, c), edgecolor='black')
            else:
                rect = ax.bar((k+1)*w, 100*acc, w, edgecolor='black')
            x = (rect[0].get_x() + rect[0].get_width()/2.)
            y = 1.03*rect[0].get_height()
            ax.text(x, y, "%.1f%%" %(100*acc), ha='center', va='bottom', fontsize=10)
            k+=1
        label = False
        k+=sep/w
        plt.gca().set_prop_cycle(None)
    ax.legend()
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True ,      # ticks along the bottom edge are on
        top=False,         # ticks along the top edge are off
        labelbottom=True,
        rotation=10) # labels along the bottom edge are off

    ax.set_xticks(pos)
    ax.set_xticklabels(tickslabels)
    ax.set(ylabel='Tasa acierto [%]')
    ax.set_ylim(ymax=100)
    fig.suptitle("ANÁLISIS MULTIPLICIDAD EN EL CONJUNTO DE DATOS " + str(conjunto))
    fig.savefig(SAVE_DIR+'bar.png')


# code.interact(local=locals())

