% Script para la extracción de características para la clasificación de valence y arousal.

% Usa: filtrarEEG.m (usa filtrosIIR.m), calcularPotencias.m

clear all;

%% Parámetros

% Características a extraer.
CONJUNTO_CARACTERISTICAS = 4;
% Se pueden extraer 3 conjuntos diferentes de características tanto para valencia como para arousal, 
% haciendo un total 6 conjuntos: 
%  - Valencia
%      Conjunto 1: Se extraen las potencias en todas las bandas (5) y electrodos y la asimetría entre
%                  14 pares de electrodos en todas las bandas. 230 características en total.
%      Conjunto 2: Se extraen las características propuestas en 'Erim Yurci. Emotion Detection from EEG
%                  Signals: Correlating Cerebral Cortex Activity with Music Evoked Emotion' para la
%                  clasificación de la valencia. 10 características en total.
%      Conjunto 3: Se extraen las potencias en las parejas banda electrodo que presentan una mayor 
%                  correlación con la valencia según el artículo correspondiente al DEAP dataset. 24
%                  características en total.
%  - Excitación
%      Conjunto 4: Se extraen las potencias en todas las bandas (5) y electrodos y la asimetría entre
%                  14 pares de electrodos en todas las bandas. 230 características en total.
%      Conjunto 2: Se extraen las características propuestas en 'Erim Yurci. Emotion Detection from EEG
%                  Signals: Correlating Cerebral Cortex Activity with Music Evoked Emotion' para la
%                  clasificación del grado de excitación. 17 características en total.
%      Conjunto 6: Se extraen las potencias en las parejas banda electrodo que presentan una mayor 
%                  correlación con el grado de excitación según el artículo correspondiente al DEAP dataset. 
%                  17 características en total.

% Las etiquetas de cada trial/vídeo son ratings de 1 a 9 dados por cada paciente.
% Para realizar posteriormente una clasificación binaria, agrupamos en dos clases.
% CLASE1: De 1 a PUNTUACION_CLASE1
PUNTUACION_CLASE1 = 3;
% CLASE2: De PUNTUACION_CLASE2 a 9
PUNTUACION_CLASE2 = 7;

% Puntuaciones test
PUNTUACION_TEST1 = 2;
PUNTUACION_TEST2 = 8;

% Puntuaciones neutras
PUNTUACION1_CLASE3 = 3.8;
PUNTUACION2_CLASE3 = 6.2;

% Puntuaciones muy neutras
PUNTUACION_N1 = 4.4;
PUNTUACION_N2 = 5.6;

% Tiempo de comienzo y fin: trozo de las señales EEG a utilizar. Se pueden indicar varios trozos
% de forma que de cada trial se obtengan múltiples muestras en lugar de una (cada trozo de la señal
% EEG se trataría como una muestra diferente).
% TSTART        = [33];             % multiplicidad 1
% TSTOP         = [63];
% TSTART        = [43 28];          % multiplicidad 2
% TSTOP         = [63 48];
TSTART      = [49 37 25];       % multiplicidad 3
TSTOP       = [63 51 39];
% TSTART      = [53 45 37 29];   % multiplicidad 4
% TSTOP       = [63 55 47 39];

multiplicidad = length(TSTART);

% Pares de electrodos simétricos a considerar.
PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];          % Hemisferio derecho.
PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];          % Hemisferio izquierdo.

% Clasificar valece o arousal
tipo = "valence";

% Directorio donde se encuentran los datos preprocesados en formato matlab.
DIR_DATASET = '/home/javi/Documents/TFM/Dataset';
% DIR_DATASET = 'C:\Users\Javi\Documents\TFM\Dataset'

% Directorio donde se guardarán los ficheros
DIR_GUARDAR = '../caracteristicas/';
% DIR_GUARDAR = '..\caracteristicas\';

% Parámetros DEAP dataset. No tocar.
Fs              = 128;  % Hz.
DURACION_EEG    = 63;   % s.
N_PACIENTES     = 32;
N_ELECTRODOS    = 32;
N_TRIALS        = 40;

% Parámetros para la extracción de las características.
% TSTART      = 0;                                                  % Tiempo de inicio para el procesado de la señal EEG.
% TSTOP       = DURACION_EEG;                                       % Tiempo de fin    para el procesado de la señal EEG.
PACIENTES   = 1:N_PACIENTES;                                        % Vector con los pacientes a procesar.
ELECTRODOS  = 1:N_ELECTRODOS;                                       % Vector con los electrodos a procesar.
TRIALS      = 1:N_TRIALS;                                           % Vector con los vídeos/trials a procesar.
% PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];        % Hemisferio derecho.
% PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];        % Hemisferio izquierdo.

%% Inicialización de estructuras de datos

switch CONJUNTO_CARACTERISTICAS
    case 1
        tipo        = "valence";
        PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];  % Hemisferio derecho.
        PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];  % Hemisferio izquierdo.
        ELECTRODOS  = 1:N_ELECTRODOS; 
    case 2
        tipo        = "valence";
        PARES       = [];
        ELECTRODOS  = [2 3 4 5 8 12 14 18 20 21 22 26 30 32];
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
    case 3
        tipo        = "valence";
        pares       = [];
        ELECTRODOS  = [2 3 8 10 14 15 17 21 22 24 25 26 27 28 30 31 32];  %TODO   
        % 2 	AF3     1
        % 3     F3      2
        % 8 	T7      3
        % 10	CP1     4
        % 14	O1      5
        % 15	Oz      6
        % 17	Fp2     7
        % 21	F8      8
        % 22	FC6     9
        % 24	Cz      10
        % 25	C4      11
        % 26	T8      12
        % 27	CP6     13
        % 28	CP2     14
        % 30	P8      15
        % 31	PO4     16
        % 32	O2      17
    case 4
        tipo        = "arousal";
        PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];  % Hemisferio derecho.
        PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];  % Hemisferio izquierdo.
        ELECTRODOS  = 1:N_ELECTRODOS; 
    case 5
        tipo        = "arousal";
        pares       = [];
        ELECTRODOS  = [2 3 4 5 8 12 14 18 20 21 22 26 30 32]; 
    case 6
        tipo        = "arousal";
        pares       = [];
        ELECTRODOS  = [1 3 4 6 7 11 17 18  21 22 23 24 27 30 31];
        % 1     Fp1     1
        % 3 	F3      2
        % 4 	F7      3
        % 6 	FC1     4
        % 7 	C3      5
        % 11	P3      6
        % 17	Fp2     7
        % 21	F8      8
        % 22	FC6     9
        % 23	FC2     10
        % 24	Cz      11
        % 27	CP6     12
        % 30	P8      13
        % 31	PO4     14
    otherwise
        ;;
end

N_DIF_PARES  = size(PARES, 2);
N_PACIENTES  = length(PACIENTES);
N_ELECTRODOS = length(ELECTRODOS);
N_TRIALS     = length(TRIALS);
N_BANDAS     = 5;

N_TRIALS_M   = N_TRIALS * multiplicidad;

fprintf("Procesando características para %d vídeos por paciente y %d pacientes en total...\n\n", N_TRIALS_M, N_PACIENTES);

% Matrices con estructuras donde se van a guardar las características y etiquetas.
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).total        = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).theta        = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).slow_alpha   = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).alpha        = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).beta         = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).gamma        = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).theta_r      = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).slow_alpha_r = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).alpha_r      = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).beta_r       = [];
potencias(N_PACIENTES, N_TRIALS_M, N_ELECTRODOS+N_DIF_PARES).gamma_r      = [];

etiquetas(N_PACIENTES, N_TRIALS_M).valence    = [];
etiquetas(N_PACIENTES, N_TRIALS_M).arousal    = [];
etiquetas(N_PACIENTES, N_TRIALS_M).dominance  = [];
etiquetas(N_PACIENTES, N_TRIALS_M).liking     = [];

% Matrices de características diferentes
potencias_theta  = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);
potencias_salpha = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);
potencias_alpha  = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);
potencias_beta   = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);
potencias_gamma  = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);
potencias_total  = zeros(N_PACIENTES*N_TRIALS_M, N_ELECTRODOS);

potencias_pares_theta  = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);
potencias_pares_salpha = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);
potencias_pares_alpha  = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);
potencias_pares_beta   = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);
potencias_pares_gamma  = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);
potencias_pares_total  = zeros(N_PACIENTES*N_TRIALS_M, N_DIF_PARES);

% Matrices etiquetas
etiquetas_valence   = zeros(N_PACIENTES*N_TRIALS_M, 1);
etiquetas_arousal   = zeros(N_PACIENTES*N_TRIALS_M, 1);
etiquetas_dominance = zeros(N_PACIENTES*N_TRIALS_M, 1);
etiquetas_liking    = zeros(N_PACIENTES*N_TRIALS_M, 1);


%% Extracción de características. 

% Por cada paciente
for p = 1:N_PACIENTES

    paciente = PACIENTES(p);

    fprintf("Paciente #%02d...\n", paciente);

    % Cargamos sus datos de EEG y etiquetas.
    fileName = sprintf('s%02d.mat', paciente);
    datos_paciente = fullfile(DIR_DATASET, fileName);
    load(datos_paciente);
    %  > 'data':   Datos EEG (40x40x8064): 40 trials/videos x (32 electrodos + 8 señales fisiologicas) x (EEG 63s a 128Hz).
    %  > 'labels': Etiquetas (40x4): 40 trials/videos x 4 ratings (valence, arousal, dominance y liking).

    % Inicializamos la matriz con las señales EEG por canal: Fs*(TSTOP-TSTART)x(N_ELECTRODOS * N_TRIALS).
    s_eeg = zeros(Fs*(TSTOP(1)-TSTART(1)), multiplicidad*N_ELECTRODOS*N_TRIALS);
    for v = 1:N_TRIALS
        video = TRIALS(v);
        % s_eeg_ = zeros(Fs*(TSTOP(1)-TSTART(1)), multiplicidad*N_ELECTRODOS);
        s_eeg_ = [];
        for m = 1:multiplicidad
            % s_eeg_(:,1+N_ELECTRODOS*(m-1):N_ELECTRODOS*m) = reshape(data(video,ELECTRODOS,1+Fs*TSTART(m):Fs*TSTOP(m)), N_ELECTRODOS, Fs*(TSTOP(m)-TSTART(m)))';
            s_eeg_ = [s_eeg_, reshape(data(video,ELECTRODOS,1+Fs*TSTART(m):Fs*TSTOP(m)), N_ELECTRODOS, Fs*(TSTOP(m)-TSTART(m)))'];
        end
        s_eeg(:,1+(v-1)*N_ELECTRODOS*multiplicidad:v*N_ELECTRODOS*multiplicidad) = s_eeg_;
    end
    % Queda una matriz de la siguiente forma:
    % - Cada fila es un instante de tiempo de una señal EEG
    % - Cada columna indica una señal EEG para un par electrodo,vídeo
    %   - Orden: [ Video 1: e1 e2 e3 ... eN, Video 2: e1 e2 e3 ... eN, Video 3: ..., ..., Video M: e1 e2 ... eN]

    % Filtrado de las señales.
    [s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma] = filtrarEEG(s_eeg, Fs);

    % Cálculo de potencias.
    [p_theta, p_salpha, p_alpha, p_beta, p_gamma, p_total] = ...
    calcularPotencias(N_ELECTRODOS, N_TRIALS_M, PARES, s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma);

    %% Guardar potencias y etiquetas en matrices de características y etiquetas

    % 0) Etiquetas
    e_v = zeros(N_TRIALS_M, 1);
    e_a = zeros(N_TRIALS_M, 1);
    for m = 1:multiplicidad
        index = m:multiplicidad:N_TRIALS_M;
        e_v(index, 1) = labels(:, 1);
        e_a(index, 1) = labels(:, 2);
        e_d(index, 1) = labels(:, 3);
        e_l(index, 1) = labels(:, 4);
    end
    etiquetas_valence(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)   = e_v;
    etiquetas_arousal(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)   = e_a;
    etiquetas_dominance(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:) = e_d;
    etiquetas_liking(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)    = e_l;

    % 1) Potencias absolutas en cada banda en cada electrodo
    potencias_theta(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_theta(1:N_ELECTRODOS*N_TRIALS_M),  N_ELECTRODOS, N_TRIALS_M)';
    potencias_salpha(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:) = reshape(p_salpha(1:N_ELECTRODOS*N_TRIALS_M), N_ELECTRODOS, N_TRIALS_M)';
    potencias_alpha(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_alpha(1:N_ELECTRODOS*N_TRIALS_M),  N_ELECTRODOS, N_TRIALS_M)';
    potencias_beta(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)   = reshape(p_beta(1:N_ELECTRODOS*N_TRIALS_M),   N_ELECTRODOS, N_TRIALS_M)';
    potencias_gamma(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_gamma(1:N_ELECTRODOS*N_TRIALS_M),  N_ELECTRODOS, N_TRIALS_M)';

    % 2) Potencia total absoluta por electrodo
    potencias_total(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_total(1:N_ELECTRODOS*N_TRIALS_M), N_ELECTRODOS, N_TRIALS_M)';

    % 3) Asimetría potencias en cada banda (pares electrodos simétricos)
    potencias_pares_theta(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_theta(1+N_ELECTRODOS*N_TRIALS_M:end),  N_DIF_PARES, N_TRIALS_M)';
    potencias_pares_salpha(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:) = reshape(p_salpha(1+N_ELECTRODOS*N_TRIALS_M:end), N_DIF_PARES, N_TRIALS_M)';
    potencias_pares_alpha(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_alpha(1+N_ELECTRODOS*N_TRIALS_M:end),  N_DIF_PARES, N_TRIALS_M)';
    potencias_pares_beta(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)   = reshape(p_beta(1+N_ELECTRODOS*N_TRIALS_M:end),   N_DIF_PARES, N_TRIALS_M)';
    potencias_pares_gamma(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_gamma(1+N_ELECTRODOS*N_TRIALS_M:end),  N_DIF_PARES, N_TRIALS_M)';

    % 4) Asímetría potencia total (pares electrodos simétricos)
    potencias_pares_total(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M,:)  = reshape(p_total(1+N_ELECTRODOS*N_TRIALS_M:end),  N_DIF_PARES, N_TRIALS_M)';

    %% Guardar potencias en estructura

    % La estructura no tiene mucho uso aparte del de consultar de forma simple las potencias/asimetría de un trial 
    
    % Por cada trial/video.
    for v = 1:N_TRIALS_M

        video = TRIALS(ceil(v/multiplicidad));

        % Estructura: guardamos las potencias de los electrodos.
        for e = 1:N_ELECTRODOS
            potencias(p,v,e).total        = p_total((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).theta        = p_theta((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).slow_alpha   = p_salpha((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).alpha        = p_alpha((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).beta         = p_beta((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).gamma        = p_gamma((v-1)*N_ELECTRODOS+e);
        end

        for d = 1:N_DIF_PARES
            potencias(p,v,N_ELECTRODOS+d).total      = p_total(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).theta      = p_theta(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).slow_alpha = p_salpha(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).alpha      = p_alpha(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).beta       = p_beta(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).gamma      = p_gamma(N_TRIALS_M*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
        end

        % Guardamos las etiquetas.
        etiquetas(p,v).valence   = labels(video, 1);
        etiquetas(p,v).arousal   = labels(video, 2);
        etiquetas(p,v).dominance = labels(video, 3);
        etiquetas(p,v).liking    = labels(video, 4);     

    end 

end

potencias_bandas        = [potencias_theta, potencias_salpha, potencias_alpha, potencias_beta, potencias_gamma];
potencias_pares_bandas  = [potencias_pares_alpha, potencias_pares_salpha, potencias_pares_alpha, potencias_pares_beta, potencias_pares_gamma];

%% Guardamos las potencias y etiquetas en ficheros .csv
fprintf('\nGuardando características en ficheros...\n')
csvwrite(strcat(DIR_GUARDAR, 'potencias_bandas.csv'),                 potencias_bandas);
csvwrite(strcat(DIR_GUARDAR, 'potencias_total.csv'),                  potencias_total);
csvwrite(strcat(DIR_GUARDAR, 'potencias_asimetria_pares_bandas.csv'), potencias_pares_bandas);
csvwrite(strcat(DIR_GUARDAR, 'potencias_asimetria_pares_total.csv'),  potencias_pares_total);

csvwrite(strcat(DIR_GUARDAR, 'etiquetas_valence.csv'),   etiquetas_valence);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_arousal.csv'),   etiquetas_arousal);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_dominance.csv'), etiquetas_dominance);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_liking.csv'),    etiquetas_liking);


%% Extracción de características y muestras.

if (tipo == "valence")
    etiquetas = etiquetas_valence;
else
    etiquetas = etiquetas_arousal;
end

CLASE1 = 0;
CLASE2 = 1;
CLASE3 = 2;

% Las muestras estarán ordenadas por paciente. Se buscan aquellas con ratings de acuerdo a los valores 
% asignados a PUNTUACION_CLASE1 y PUNTUACION_CLASE2: Se guardan los índices en muestras_i.
muestras_i   = [];
tags         = [];
t_muestras_i = [];
t_tags       = [];
n_muestras_i = [];
n_tags       = [];
n_muestras_paciente = zeros(N_PACIENTES, 1);
for p = 1:N_PACIENTES
    paciente  = PACIENTES(p);
    % Muestras más significativas para test
    test1_i   = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<=PUNTUACION_TEST1);
    test2_i   = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>=PUNTUACION_TEST2);
    [notest1_i, n1_i] = datasample(test1_i, round(length(test1_i)*2/3), 1, 'Replace', false);
    test1_i(n1_i,:)   = [];
    [notest2_i, n2_i] = datasample(test2_i, round(length(test2_i)*2/3), 1, 'Replace', false);
    test2_i(n2_i,:)   = [];

    % Muestras más neutras para test
    test3_i = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>=PUNTUACION_N1 & ...
                                      etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<=PUNTUACION_N2);
    [notest3_i, n3_i] = datasample(test3_i, round(length(test3_i)*2/3), 1, 'Replace', false);
    test3_i(n3_i,:)   = [];

    % Muestras para entrenamiento
    c1_i      = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<=PUNTUACION_CLASE1 & ...
                                        etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>PUNTUACION_TEST1); 
    if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
        c2_i  = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>PUNTUACION_CLASE2 & ...
                                        etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<PUNTUACION_TEST2);
    else 
        c2_i  = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>=PUNTUACION_CLASE2 & ...
                                        etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<PUNTUACION_TEST2);
    end
    c3_1_i    = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>=PUNTUACION1_CLASE3 & ...
                                        etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<PUNTUACION_N1); 
    c3_2_i    = (p-1)*N_TRIALS_M + find(etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)<=PUNTUACION2_CLASE3 & ...
                                        etiquetas(1+(p-1)*N_TRIALS_M:p*N_TRIALS_M)>PUNTUACION_N2);
    c1_i = [c1_i; notest1_i];
    c2_i = [c2_i; notest2_i];
    c3_i = [c3_1_i; c3_2_i; notest3_i];
    
    fprintf("Paciente %d %s: \n", paciente, tipo);
    n_muestras_paciente(p) = length(c1_i) + length(c2_i);
    fprintf("  Train: %d (0) y %d (1)\n", length(c1_i), length(c2_i));
    muestras_i = [muestras_i; c1_i;                            c2_i;                             c3_i];
    tags       = [tags;       repmat(CLASE1, length(c1_i), 1);  repmat(CLASE2, length(c2_i), 1); repmat(CLASE3, length(c3_i), 1)];

    fprintf("  Test: %d (0) y %d (1)\n", length(test1_i), length(test2_i));
    t_muestras_i = [t_muestras_i; test1_i;                             test2_i;                            test3_i];
    t_tags       = [t_tags;       repmat(CLASE1, length(test1_i), 1);  repmat(CLASE2, length(test2_i), 1); repmat(CLASE3, length(test3_i), 1)];

end
n_muestras = sum(n_muestras_paciente);

todo_i = [t_muestras_i; muestras_i];

% Características 
switch CONJUNTO_CARACTERISTICAS
    case 1
        features = [potencias_bandas(todo_i,:), potencias_pares_bandas(todo_i,:)];
    case 2
        features         = zeros(n_muestras, 10);
        features(:,1)    = (potencias_beta(todo_i,8)  - potencias_alpha(todo_i,8)) ./ (potencias_beta(todo_i, 1) - potencias_alpha(todo_i,1));
        features(:,2)    = ((potencias_beta(todo_i,8) + potencias_beta(todo_i, 9)) ./ (potencias_alpha(todo_i,8) + potencias_alpha(todo_i,9)))  - ...
                           ((potencias_beta(todo_i,1) + potencias_beta(todo_i, 2)) ./ (potencias_alpha(todo_i,1) + potencias_alpha(todo_i,2)));
        features(:,3)    = (sum(potencias_beta(todo_i, 8:14), 2) ./ sum(potencias_alpha(todo_i, 8:14), 2))  - ...
                           (sum(potencias_beta(todo_i, 1:7),  2) ./ sum(potencias_alpha(todo_i, 1:7),  2));
        features(:,4:10) = potencias_total(todo_i,8:14) - potencias_total(todo_i,1:7);
    case 3
        features = [potencias_theta(todo_i, [5 6 17 16 7]), ...
                    potencias_alpha(todo_i, [2 1 5 6 16 13]), ...
                    potencias_beta(todo_i,  [3 4 10 6 9]), ...
                    potencias_gamma(todo_i, [3 14 11 13 9 15 8 12])];
    case 4
        features = [potencias_bandas(todo_i,:), potencias_pares_bandas(todo_i,:)];
    case 5
        features         = zeros(n_muestras, 17);
        features(:,1)    = (potencias_beta(todo_i,8) + potencias_beta(todo_i,1)) ./ (potencias_alpha(todo_i,8) + potencias_alpha(todo_i,1));
        features(:,2)    = sum(potencias_beta(todo_i,[8 1 9 2]), 2) ./ sum(potencias_alpha(todo_i,[8 1 9 2]), 2);
        features(:,3)    = sum(potencias_beta(todo_i,8:14), 2) ./ sum(potencias_alpha(todo_i,8:14), 2);
        features(:,4:17) = potencias_beta(todo_i,  [1 8 2 9 3 10 4 11 5 12 6 13 7 14]) - ...
                           potencias_alpha(todo_i, [1 8 2 9 3 10 4 11 5 12 6 13 7 14]);
    case 6
        features = [potencias_theta(todo_i, [11 7 9 12]), ...
                    potencias_alpha(todo_i, [3 5 1 11]), ...
                    potencias_beta(todo_i,  [3 1 10 8]), ...
                    potencias_gamma(todo_i, [2 4 7 14 13])];   
    otherwise
        features = [potencias_bandas(todo_i,:), potencias_pares_bandas(todo_i,:)];
end

test_features  = features(1:length(t_muestras_i),:);
train_features = features(1+length(t_muestras_i):end, :);

% Guardar dataset en fichero .csv
file_dataset   = DIR_GUARDAR + "dataset" + CONJUNTO_CARACTERISTICAS + "_m" + multiplicidad + "_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_dataset, [features, [t_tags; n_tags; tags]]);

file_dataset_test   = DIR_GUARDAR + "dataset" + CONJUNTO_CARACTERISTICAS + "_test_m" + multiplicidad + "_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_dataset_test, [test_features, t_tags]);

file_dataset_train   = DIR_GUARDAR + "dataset" + CONJUNTO_CARACTERISTICAS + "_train_m" + multiplicidad + "_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_dataset_train, [train_features, tags]);

file_nmuestras = DIR_GUARDAR + "n_muestras_" + tipo + "_paciente_m" + multiplicidad + "_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_nmuestras, n_muestras_paciente);

