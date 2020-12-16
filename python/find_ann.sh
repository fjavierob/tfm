#!/bin/bash

ENV_="tfm36"
PYTHON_FILE="clasificador_ann.py"
RESULTS_TMP="./ann_tmp"
HIGH_ACC=75


# DATASET_DIR='../caracteristicas/'
# DATASET_NAME='dataset2_m1_3_7.csv'
# DATASET_FILE="${DATASET_DIR}""${DATASET_NAME}"

DATASET_NAME=$(echo ${DATASET_FILE##*/} | cut -f1 -d.)

# SAVE_BASE="./resultados/"
SAVE_DIR="${SAVE_BASE}ann/${DATASET_NAME}"
RESULTS_FILE="${SAVE_BASE}ann/resultados.txt"

# Parametros ANN
SEED_MIN=0
SEED_MAX=15
SEED_RANGE=$(($SEED_MAX-$SEED_MIN+1));
BATCH_SIZE="30 50 80"
N_EPOCHS="50 100 150 200 250"

ITERACIONES=25

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_" ]]; then
    activate $ENV_
fi
if [ ! -d "$SAVE_DIR" ]; then
  mkdir -p $SAVE_DIR
fi

echo "" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "          ${DATASET_FILE##*/} $(date)" >> $RESULTS_FILE
echo "------------------------------------------------------------------------------------------------------" >> $RESULTS_FILE
for BATCH in $BATCH_SIZE; do
    echo "Batch ${BATCH}:"
    BATCH_DIR="${SAVE_DIR}batch${BATCH}"
    if [ ! -d "$BATCH_DIR" ]; then
        mkdir $BATCH_DIR
    fi
    for EPOCHS in $N_EPOCHS; do
        echo "  Epoch $EPOCHS"
        for i in $(eval echo "{1..$ITERACIONES}"); do
            if [[ i -eq 1 ]] || [[ $(($i % 5)) -eq 0 ]]; then
                echo -n "$i "
            fi
            # Obtener semilla aleatoria
            SEED=$RANDOM
            let "SEED %= $SEED_RANGE";
            SEED=$(($SEED+$SEED_MIN));
            # Ruta al fichero del modelo de la ann
            MODEL_PATH="${BATCH_DIR}""/model_""${EPOCHS}""_""${SEED}"
            # Ejecutar ann
            python $PYTHON_FILE $DATASET_FILE $SEED $BATCH $EPOCHS $RESULTS_TMP $MODEL_PATH > /dev/null 2>&1
            RESULTADO_VAL=$(cat $RESULTS_TMP | cut -f1 -d,)
            RESULTADO_ALL=$(cat $RESULTS_TMP | cut -f2 -d,) 
            RESULTADO_VAL_FLOOR="${RESULTADO_VAL:0:2}"
            RESULTADO_ALL_FLOOR="${RESULTADO_ALL:0:2}"
            if [[ RESULTADO_VAL_FLOOR -ge HIGH_ACC ]]; then
                if [[ RESULTADO_ALL_FLOOR -ge HIGH_ACC ]]; then
                    echo "[!!!]  ${RESULTADO_VAL} - ${RESULTADO_ALL} : ${MODEL_PATH}" >> $RESULTS_FILE
                else
                    echo "[!]    ${RESULTADO_VAL} - ${RESULTADO_ALL} : ${MODEL_PATH}" >> $RESULTS_FILE
                fi
            else
                echo "       $MODEL_PATH : ${RESULTADO_VAL} - ${RESULTADO_ALL}" >> $RESULTS_FILE
            fi
        done
        echo ""
    done
done