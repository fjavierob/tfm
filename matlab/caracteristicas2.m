% Script para la extracción de características para la clasificación de valence
% y arousal usadas en el paper Erim-Yurci-Master-Thesis-2014.pdf

% Usa: filtros_iir.m, calcularPotencias2.m

clear all;

%% Parámetros

% Directorio donde se encuentran los datos del DEAPdataset preprocesados en formato matlab.
DIR_DATASET = '/home/javi/Documents/TFM/Dataset';

% Directorio donde se guardarán los ficheros
DIR_GUARDAR = '../caracteristicas/';

% Las etiquetas de cada trial/vídeo son ratings de 1 a 9 dados por cada paciente.
% Para realizar posteriormente una clasificación binaria, agrupamos en dos clases.
% CLASE1: De 1 a PUNTUACION_CLASE1
PUNTUACION_CLASE1 = 3;
% CLASE2: De PUNTUACION_CLASE2 a 9
PUNTUACION_CLASE2 = 7;

% Parámetros DEAP dataset.
Fs              = 128;  % Hz.
DURACION_EEG    = 63;   % s.
N_PACIENTES     = 32;
N_ELECTRODOS    = 32;
N_TRIALS        = 40;

% Parámetros por defecto para la extracción de las características.
TSTART      = 0;                                                    % Tiempo de inicio para el procesado de la señal EEG.
TSTOP       = DURACION_EEG;                                         % Tiempo de fin    para el procesado de la señal EEG.
PACIENTES   = 1:N_PACIENTES;                                        % Vector con los pacientes a procesar.
ELECTRODOS  = 1:N_ELECTRODOS;                                       % Vector con los electrodos a procesar.
TRIALS      = 1:N_TRIALS;                                           % Vector con los vídeos/trials a procesar.
PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];          % Hemisferio derecho.
PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];          % Hemisferio izquierdo.
[f_theta, f_alpha, f_beta, f_gamma] = filtros_iir(Fs);              % Filtros para las diferentes bandas.

% Editar parámetros aquí
PARES       = [];
TSTART      = 33;
TSTOP       = 63;
ELECTRODOS  = [2 3 4 5 8 12 14 18 20 21 22 26 30 32];   % Electrodos usados en Erim-Yurci-Master-Thesis-2014.pdf
% 2     AF3     1
% 3 	F3      2
% 4 	F7      3
% 5 	FC5     4
% 8 	T7      5
% 12	P7      6
% 14	O1      7
% 18	AF4     8
% 20	F4      9
% 21	F8      10
% 22	FC6     11
% 26	T8      12
% 30	P8      13
% 32	O2      14


%% Inicialización de estructuras de datos

N_DIF_PARES  = size(PARES, 2);
N_PACIENTES  = length(PACIENTES);
N_ELECTRODOS = length(ELECTRODOS);
N_TRIALS     = length(TRIALS);
N_BANDAS     = 3;

% Estructuras donde se van a guardar las características y etiquetas.
potencias_theta = zeros(N_PACIENTES*N_TRIALS, N_ELECTRODOS);
potencias_alpha = zeros(N_PACIENTES*N_TRIALS, N_ELECTRODOS);
potencias_beta  = zeros(N_PACIENTES*N_TRIALS, N_ELECTRODOS);
potencias_total = zeros(N_PACIENTES*N_TRIALS, N_ELECTRODOS);

potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).total        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).theta        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).alpha        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).beta         = [];

etiquetas(N_PACIENTES, N_TRIALS).valence    = [];
etiquetas(N_PACIENTES, N_TRIALS).arousal    = [];

% Matrices etiquetas
etiquetas_valence   = zeros(N_PACIENTES*N_TRIALS, 1);
etiquetas_arousal   = zeros(N_PACIENTES*N_TRIALS, 1);


%% Extracción de características. 

% Por cada paciente
for p = 1:N_PACIENTES

    paciente = PACIENTES(p);

    % fprintf("Paciente #%02d...\n", paciente);
    fprintf("%d ", paciente);

    % Cargamos sus datos de EEG y etiquetas.
    fileName = sprintf('s%02d.mat', paciente);
    datos_paciente = fullfile(DIR_DATASET, fileName);
    load(datos_paciente);
    %  > 'data':   Datos EEG (40x40x8064): 40 trials/videos x (32 electrodos + 8 señales fisiologicas) x (EEG 63s a 128Hz).
    %  > 'labels': Etiquetas (40x4): 40 trials/videos x 4 ratings (valence, arousal, dominance y liking).

    % Inicializamos la matriz con las señales EEG por canal: Fs*(TSTOP-TSTART)x(N_ELECTRODOS * N_TRIALS).
    s_eeg = zeros(Fs*(TSTOP-TSTART), N_ELECTRODOS*N_TRIALS);
    for v = 1:N_TRIALS
        video = TRIALS(v);
        s_eeg(:,1+(v-1)*N_ELECTRODOS:v*N_ELECTRODOS) = reshape(data(video,ELECTRODOS,1+Fs*TSTART:Fs*TSTOP), N_ELECTRODOS, Fs*(TSTOP-TSTART))';
    end
    % Queda una matriz de la siguiente forma:
    % - Cada fila es un instante de tiempo de una señal EEG
    % - Cada columna indica una señal EEG para un par electrodo,vídeo
    %   - Orden: [ Video 1: e1 e2 e3 ... eN, Video 2: e1 e2 e3 ... eN, Video 3: ..., ..., Video M: e1 e2 ... eN]

    % Filtrado de las señales.
    s_eeg_theta     = filter(f_theta,   s_eeg);
    s_eeg_alpha     = filter(f_alpha,   s_eeg);
    s_eeg_beta      = filter(f_beta,    s_eeg);
    s_eeg_gamma     = filter(f_gamma,   s_eeg);

    % Cálculo de potencias.
    [p_theta, p_alpha, p_beta, p_total] = calcularPotencias2(N_ELECTRODOS, N_TRIALS, PARES, s_eeg_theta, s_eeg_alpha, s_eeg_beta, s_eeg_gamma);

    %% Guardar potencias y etiquetas en matrices de características y etiquetas

    % Dividimos en dos clases (0, 1) las etiquetas (ratings de 1 a 9)
    % labels = labels > 5;

    % 0) Etiquetas
    etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS,:)   = labels(:, 1);
    etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS,:)   = labels(:, 2);

    % 1) Potencias absolutas en cada banda (por electrodo)
                                                 % Reshape: Matriz potencia en banda: videos x electrodos    
    
    potencias_theta(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = reshape(p_theta(1:N_ELECTRODOS*N_TRIALS), N_ELECTRODOS, N_TRIALS)';
    potencias_alpha(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = reshape(p_alpha(1:N_ELECTRODOS*N_TRIALS), N_ELECTRODOS, N_TRIALS)';
    potencias_beta(1+(p-1)*N_TRIALS:p*N_TRIALS,:)  = reshape(p_beta(1:N_ELECTRODOS*N_TRIALS),  N_ELECTRODOS, N_TRIALS)';

    % 2) Potencia total absoluta (por electrodo)
    potencias_total(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = reshape(p_total(1:N_ELECTRODOS*N_TRIALS), N_ELECTRODOS, N_TRIALS)';

    %% Guardar potencias en estructura. 
    
    % La estructura no tiene mucho uso aparte del de consultar de forma simple las potencias de un trial.
    
    % Por cada trial/video.
    for v = 1:N_TRIALS
        video = TRIALS(v);
        % Estructura: guardamos las potencias de los electrodos.
        for e = 1:N_ELECTRODOS
            potencias(p,v,e).total        = p_total((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).theta        = p_theta((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).alpha        = p_alpha((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).beta         = p_beta((v-1)*N_ELECTRODOS+e);
        end
        % Guardamos las etiquetas.
        etiquetas(p,v).valence   = labels(video, 1);
        etiquetas(p,v).arousal   = labels(video, 2); 
    end 
    
end
fprintf('\n');

%% Características paper Erim-Yurci-Master-Thesis-2014.pdf

% Procedemos a calcular las características utilizadas en el paper anterior.

% Características Valence
c_valence = zeros(N_PACIENTES*N_TRIALS, 10);

c_valence(:,1)      = (potencias_beta(:,8)  - potencias_alpha(:,8)) ./ (potencias_beta(:, 1) - potencias_alpha(:,1));
c_valence(:,2)      = ((potencias_beta(:,8) + potencias_beta(:, 9)) ./ (potencias_alpha(:,8) + potencias_alpha(:,9)))  - ...
                      ((potencias_beta(:,1) + potencias_beta(:, 2)) ./ (potencias_alpha(:,1) + potencias_alpha(:,2)));
c_valence(:,3)      = (sum(potencias_beta(:, 8:14), 2) ./ sum(potencias_alpha(:, 8:14), 2))  - ...
                      (sum(potencias_beta(:, 1:7), 2)  ./ sum(potencias_alpha(:, 1:7), 2));
c_valence(:,4:10)   = potencias_total(:,8:14) - potencias_total(:,1:7);

% Características Arousal
c_arousal = zeros(N_PACIENTES*N_TRIALS, 17);

c_arousal(:,1)      = (potencias_beta(:,8) + potencias_beta(:,1)) ./ (potencias_alpha(:,8) + potencias_alpha(:,1));
c_arousal(:,2)      = sum(potencias_beta(:,[8 1 9 2]), 2) ./ sum(potencias_alpha(:,[8 1 9 2]), 2);
c_arousal(:,3)      = sum(potencias_beta(:,8:14), 2) ./ sum(potencias_alpha(:,8:14), 2);
c_arousal(:,4:17)   = potencias_beta(:,  [1 8 2 9 3 10 4 11 5 12 6 13 7 14]) - ...
                      potencias_alpha(:, [1 8 2 9 3 10 4 11 5 12 6 13 7 14]);


%% Guardamos las características y etiquetas en ficheros .csv
csvwrite(strcat(DIR_GUARDAR,"caracteristicas_valence.csv"), c_valence);
csvwrite(strcat(DIR_GUARDAR,"caracteristicas_arousal.csv"), c_arousal);
csvwrite(strcat(DIR_GUARDAR,"etiquetas_valence.csv"),       etiquetas_valence);
csvwrite(strcat(DIR_GUARDAR,"etiquetas_arousal.csv"),       etiquetas_arousal);


%% Datasets .csv preparado con características + etiquetas agrupadas en 2 clases.

% Etiquetas de 1 a 9: CLASE1: De 1 a PUNTUACION_CLASE1; CLASE2: De PUNTUACION_CLASE2 a 9.
CLASE1 = 0;
CLASE2 = 1;


% Valence
ntrials_paciente_v = []
fv = [];
tv = [];
for p = 1:N_PACIENTES
    paciente = PACIENTES(p);
    c1_i    = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)<=PUNTUACION_CLASE1); 
    if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
        c2_i = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)>PUNTUACION_CLASE2);
    else 
        c2_i = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)>=PUNTUACION_CLASE2);
    end
    fprintf("Paciente %d valence: %d (0) y %d (1)\n", paciente, length(c1_i), length(c2_i));
    fv_     = [c_valence(c1_i,:)        ; c_valence(c2_i,:)];
    tv_     = [etiquetas_valence(c1_i)  ; etiquetas_valence(c2_i)];
    ntrials_paciente_v(p) = length(tv_);
    fv      = [fv; fv_];
    tv      = [tv; tv_];
end
tv(tv<=PUNTUACION_CLASE1) = CLASE1; 
tv(tv>=PUNTUACION_CLASE2) = CLASE2;
dataset_v = [fv tv];
file_v    = DIR_GUARDAR + "dataset2_valence_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_v, dataset_v);

% c1_i    = find(etiquetas_valence<=PUNTUACION_CLASE1); 
% if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
%     c2_i = find(etiquetas_valence>PUNTUACION_CLASE2);
% else 
%     c2_i = find(etiquetas_valence>=PUNTUACION_CLASE2);
% end
% fv      = [c_valence(c1_i,:)        ; c_valence(c2_i,:)];
% tv      = [etiquetas_valence(c1_i)  ; etiquetas_valence(c2_i)];
% tv(tv<=PUNTUACION_CLASE1) = CLASE1; 
% tv(tv>=PUNTUACION_CLASE2) = CLASE2;
% dataset_v = [fv tv];
% file_v    = DIR_GUARDAR + "dataset_valence_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
% csvwrite(file_v, dataset_v);

% Arousal
ntrials_paciente_a = []
fa = [];
ta = [];
for p = 1:N_PACIENTES
    paciente = PACIENTES(p);
    c1_i    = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)<=PUNTUACION_CLASE1); 
    if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
        c2_i = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)>PUNTUACION_CLASE2);
    else 
        c2_i = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)>=PUNTUACION_CLASE2);
    end
    fprintf("Paciente %d arousal: %d (0) y %d (1)\n", paciente, length(c1_i), length(c2_i));
    fa_     = [c_arousal(c1_i,:)        ; c_arousal(c2_i,:)];
    ta_     = [etiquetas_arousal(c1_i)  ; etiquetas_arousal(c2_i)];
    ntrials_paciente_a(p) = length(ta_);
    fa      = [fa; fa_];
    ta      = [ta; ta_];
end
ta(ta<=PUNTUACION_CLASE1) = CLASE1; 
ta(ta>=PUNTUACION_CLASE2) = CLASE2;
dataset_a = [fa ta];
file_a    = DIR_GUARDAR + "dataset2_arousal_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_a, dataset_a);

file_ntrials     = DIR_GUARDAR + "ntrials_valence_paciente_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_ntrials, ntrials_paciente_v');
file_ntrials     = DIR_GUARDAR + "ntrials_arousal_paciente_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_ntrials, ntrials_paciente_a');

% c1_i     = find(etiquetas_arousal<=PUNTUACION_CLASE1); 
% if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
%     c2_i = find(etiquetas_arousal>PUNTUACION_CLASE2);
% else 
%     c2_i = find(etiquetas_arousal>=PUNTUACION_CLASE2);
% end    
% fa      = [c_arousal(c1_i,:)        ; c_arousal(c2_i,:)];
% ta      = [etiquetas_arousal(c1_i)  ; etiquetas_arousal(c2_i)];
% ta(ta<=PUNTUACION_CLASE1) = CLASE1; 
% ta(ta>=PUNTUACION_CLASE2) = CLASE2;
% dataset_a = [fa ta];
% file_a    = DIR_GUARDAR + "dataset_arousal_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
% csvwrite(file_a, dataset_a);


%% Comprobar el correcto cálculo de las características: 

% Comparamos el valor calculado para algunas de ellas con el valor calculado manualmente.

% c_valence(2*40+14,1) - ((potencias(3,14,8).beta - potencias(3,14,8).alpha) / (potencias(3,14,1).beta - potencias(3,14,1).alpha))

% c_valence(4*40+5,2)  - ( ( (potencias(5,5,8).beta + potencias(5,5,9).beta) / (potencias(5,5,8).alpha + potencias(5,5,9).alpha) ) - ...
%                          ( (potencias(5,5,1).beta + potencias(5,5,2).beta) / (potencias(5,5,1).alpha + potencias(5,5,2).alpha) ) )

% bet1 = 0; bet2 = 0; alph1 = 0; alph2 = 0;
% for e = 8:14
%     bet1  = bet1  + potencias(21,29,e).beta;
%     bet2  = bet2  + potencias(21,29,e-7).beta;
%     alph1 = alph1 + potencias(21,29,e).alpha;
%     alph2 = alph2 + potencias(21,29,e-7).alpha;
% end
% c_valence(20*40+29,3) - ( bet1/alph1 - bet2/alph2 )

% c_valence(30*40+40,6) - (potencias(31,40,10).total - potencias(31,40,3).total)

% c_arousal(16*40+17,1) - ( (potencias(17,17,8).beta + potencias(17,17,1).beta) / (potencias(17,17,8).alpha + potencias(17,17,1).alpha) )

% c_arousal(3*40+12,2) - ( (potencias(4,12,8).beta + potencias(4,12,1).beta + potencias(4,12,9).beta + potencias(4,12,2).beta) / ...
%                          (potencias(4,12,8).alpha + potencias(4,12,1).alpha + potencias(4,12,9).alpha + potencias(4,12,2).alpha) )

% bet = 0; alph = 0;
% for e = 8:14
%     bet  = bet  + potencias(25,3,e).beta;
%     alph = alph + potencias(25,3,e).alpha;
% end
% c_arousal(24*40+3,3) - (bet/alph)

% c_arousal(32*40,10) - (potencias(32,40,4).beta - potencias(32,40,4).alpha)


