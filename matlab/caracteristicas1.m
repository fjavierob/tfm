% Script para la extracción de características para la clasificación de valence
% y arousal propuestas en el paper tac_special_issue_2011.pdf relativo al DEAPdataset.

% Usa: filtros_iir.m, calcularPotencias1.m

clear all;

%% Parámetros

% Directorio donde se encuentran los datos preprocesados en formato matlab.
DIR_DATASET = '/home/javi/Documents/TFM/Dataset';
DIR_DATASET = 'C:\Users\Javi\Documents\TFM\Dataset'

% Directorio donde se guardarán los ficheros
DIR_GUARDAR = '../caracteristicas/';
DIR_GUARDAR = '..\caracteristicas\';

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

% Parámetros para la extracción de las características.
TSTART      = 0;                                                    % Tiempo de inicio para el procesado de la señal EEG.
TSTOP       = DURACION_EEG;                                         % Tiempo de fin    para el procesado de la señal EEG.
PACIENTES   = 1:N_PACIENTES;                                        % Vector con los pacientes a procesar.
ELECTRODOS  = 1:N_ELECTRODOS;                                       % Vector con los electrodos a procesar.
TRIALS      = 1:N_TRIALS;                                           % Vector con los vídeos/trials a procesar.
PARES(1,:)  = [17 18 20 21 23 22 25 26 28 27 29 30 31 32];          % Hemisferio derecho.
PARES(2,:)  = [1  2  3  4  6  5  7  8  10 9  11 12 13 14];          % Hemisferio izquierdo.
[f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_iir(Fs);    % Filtros para las diferentes bandas.

% Editar parámetros aquí
% PACIENTES = 1;
TSTART = 33;
TSTOP  = 63;
% TRIALS = 1:3;
% clear PARES;
% PARES(1,:)  = [17 18 20 21];          % Hemisferio derecho.
% PARES(2,:)  = [1  2  3  4 ];          % Hemisferio izquierdo.

%% Inicialización de estructuras de datos

N_DIF_PARES  = size(PARES, 2);
N_PACIENTES  = length(PACIENTES);
N_ELECTRODOS = length(ELECTRODOS);
N_TRIALS     = length(TRIALS);
N_BANDAS     = 5;

fprintf("Procesando caracteristicas para %d videos por paciente y %d pacientes en total...\n\n", N_TRIALS, N_PACIENTES);

% Matrices con estructuras donde se van a guardar las características y etiquetas.
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).total        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).theta        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).slow_alpha   = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).alpha        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).beta         = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).gamma        = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).theta_r      = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).slow_alpha_r = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).alpha_r      = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).beta_r       = [];
potencias(N_PACIENTES, N_TRIALS, N_ELECTRODOS+N_DIF_PARES).gamma_r      = [];

etiquetas(N_PACIENTES, N_TRIALS).valence    = [];
etiquetas(N_PACIENTES, N_TRIALS).arousal    = [];
etiquetas(N_PACIENTES, N_TRIALS).dominance  = [];
etiquetas(N_PACIENTES, N_TRIALS).liking     = [];

% Matrices de características diferentes
potencias_absolutas_bandas = zeros(N_PACIENTES*N_TRIALS, N_BANDAS*N_ELECTRODOS);
potencias_absolutas_total  = zeros(N_PACIENTES*N_TRIALS, N_ELECTRODOS);
potencias_relativas_bandas = zeros(N_PACIENTES*N_TRIALS, N_BANDAS*N_ELECTRODOS);
potencias_asimetria_bandas = zeros(N_PACIENTES*N_TRIALS, N_DIF_PARES*N_BANDAS);
potencias_asimetria_total  = zeros(N_PACIENTES*N_TRIALS, N_DIF_PARES);

% Matrices etiquetas
etiquetas_valence   = zeros(N_PACIENTES*N_TRIALS, 1);
etiquetas_arousal   = zeros(N_PACIENTES*N_TRIALS, 1);
etiquetas_dominance = zeros(N_PACIENTES*N_TRIALS, 1);
etiquetas_liking    = zeros(N_PACIENTES*N_TRIALS, 1);


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
    s_eeg_salpha    = filter(f_salpha,  s_eeg);
    s_eeg_alpha     = filter(f_alpha,   s_eeg);
    s_eeg_beta      = filter(f_beta,    s_eeg);
    s_eeg_gamma     = filter(f_gamma,   s_eeg);

    % Cálculo de potencias.
    % [p_theta, p_salpha, p_alpha, p_beta, p_gamma] = calcularPotencias1(N_ELECTRODOS, N_TRIALS, PARES, s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma);
    [p_theta, p_theta_r, p_salpha, p_salpha_r, p_alpha, p_alpha_r, p_beta, p_beta_r, p_gamma, p_gamma_r, p_total] = ...
    calcularPotencias1(N_ELECTRODOS, N_TRIALS, PARES, s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma);

    %% Guardar potencias y etiquetas en matrices de características y etiquetas

    % Dividimos en dos clases (0, 1) las etiquetas (ratings de 1 a 9)
    % labels = labels > 5;

    % 0) Etiquetas
    etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS,:)   = labels(:, 1);
    etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS,:)   = labels(:, 2);
    etiquetas_dominance(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = labels(:, 3);
    etiquetas_liking(1+(p-1)*N_TRIALS:p*N_TRIALS,:)    = labels(:, 4);

    % 1) Potencias absolutas en cada banda (por electrodo)
    theta_i  = 1:N_BANDAS:N_ELECTRODOS*N_BANDAS;
    salpha_i = 2:N_BANDAS:N_ELECTRODOS*N_BANDAS;
    alpha_i  = 3:N_BANDAS:N_ELECTRODOS*N_BANDAS;
    beta_i   = 4:N_BANDAS:N_ELECTRODOS*N_BANDAS;
    gamma_i  = 5:N_BANDAS:N_ELECTRODOS*N_BANDAS;
                                                                     % Reshape: Matriz potencia en banda: videos x electrodos          
    potencias_absolutas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,theta_i)  = reshape(p_theta(1:N_ELECTRODOS*N_TRIALS),  N_ELECTRODOS, N_TRIALS)';
    potencias_absolutas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,salpha_i) = reshape(p_salpha(1:N_ELECTRODOS*N_TRIALS), N_ELECTRODOS, N_TRIALS)';
    potencias_absolutas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,alpha_i)  = reshape(p_alpha(1:N_ELECTRODOS*N_TRIALS),  N_ELECTRODOS, N_TRIALS)';
    potencias_absolutas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,beta_i)   = reshape(p_beta(1:N_ELECTRODOS*N_TRIALS),   N_ELECTRODOS, N_TRIALS)';
    potencias_absolutas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,gamma_i)  = reshape(p_gamma(1:N_ELECTRODOS*N_TRIALS),  N_ELECTRODOS, N_TRIALS)';

    % 2) Potencia total absoluta en (por electrodo)
    potencias_absolutas_total(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = reshape(p_total(1:N_ELECTRODOS*N_TRIALS), N_ELECTRODOS, N_TRIALS)';

    % 3) Potencias relativas en cada banda (por electrodo)
    potencias_relativas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,theta_i)  = reshape(p_theta_r,  N_ELECTRODOS, N_TRIALS)';
    potencias_relativas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,salpha_i) = reshape(p_salpha_r, N_ELECTRODOS, N_TRIALS)';
    potencias_relativas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,alpha_i)  = reshape(p_alpha_r,  N_ELECTRODOS, N_TRIALS)';
    potencias_relativas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,beta_i)   = reshape(p_beta_r,   N_ELECTRODOS, N_TRIALS)';
    potencias_relativas_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS,gamma_i)  = reshape(p_gamma_r,  N_ELECTRODOS, N_TRIALS)';

    % 4) Asimetría potencias en cada banda (pares electrodos simétricos)
    theta_i  = 1:N_BANDAS:N_DIF_PARES*N_BANDAS;
    salpha_i = 2:N_BANDAS:N_DIF_PARES*N_BANDAS;
    alpha_i  = 3:N_BANDAS:N_DIF_PARES*N_BANDAS;
    beta_i   = 4:N_BANDAS:N_DIF_PARES*N_BANDAS;
    gamma_i  = 5:N_BANDAS:N_DIF_PARES*N_BANDAS;

    potencias_asimetria_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS, theta_i)  = reshape(p_theta(1+N_ELECTRODOS*N_TRIALS:end),  N_DIF_PARES, N_TRIALS)';
    potencias_asimetria_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS, salpha_i) = reshape(p_salpha(1+N_ELECTRODOS*N_TRIALS:end), N_DIF_PARES, N_TRIALS)';
    potencias_asimetria_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS, alpha_i)  = reshape(p_alpha(1+N_ELECTRODOS*N_TRIALS:end),  N_DIF_PARES, N_TRIALS)';
    potencias_asimetria_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS, beta_i)   = reshape(p_beta(1+N_ELECTRODOS*N_TRIALS:end),   N_DIF_PARES, N_TRIALS)';
    potencias_asimetria_bandas(1+(p-1)*N_TRIALS:p*N_TRIALS, gamma_i)  = reshape(p_gamma(1+N_ELECTRODOS*N_TRIALS:end),  N_DIF_PARES, N_TRIALS)';

    % 5) Asímetría potencia total (pares electrodos simétricos)
    potencias_asimetria_total(1+(p-1)*N_TRIALS:p*N_TRIALS,:) = reshape(p_total(1+N_ELECTRODOS*N_TRIALS:end), N_DIF_PARES, N_TRIALS)';


    %% Guardar potencias en estructura

    % La estructura no tiene mucho uso aparte del de consultar de forma simple las potencias/asimetría de un trial 
    
    % Por cada trial/video.
    for v = 1:N_TRIALS

        video = TRIALS(v);

        % Estructura: guardamos las potencias de los electrodos.
        for e = 1:N_ELECTRODOS
            potencias(p,v,e).total        = p_total((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).theta        = p_theta((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).slow_alpha   = p_salpha((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).alpha        = p_alpha((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).beta         = p_beta((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).gamma        = p_gamma((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).theta_r      = p_theta_r((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).slow_alpha_r = p_salpha_r((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).alpha_r      = p_alpha_r((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).beta_r       = p_beta_r((v-1)*N_ELECTRODOS+e);
            potencias(p,v,e).gamma_r      = p_gamma_r((v-1)*N_ELECTRODOS+e);
        end

        for d = 1:N_DIF_PARES
            potencias(p,v,N_ELECTRODOS+d).total      = p_total(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).theta      = p_theta(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).slow_alpha = p_salpha(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).alpha      = p_alpha(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).beta       = p_beta(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
            potencias(p,v,N_ELECTRODOS+d).gamma      = p_gamma(N_TRIALS*N_ELECTRODOS+(v-1)*N_DIF_PARES+d);
        end

        % Guardamos las etiquetas.
        etiquetas(p,v).valence   = labels(video, 1);
        etiquetas(p,v).arousal   = labels(video, 2);
        etiquetas(p,v).dominance = labels(video, 3);
        etiquetas(p,v).liking    = labels(video, 4);     

    end 
    
end


%% Guardamos las características y etiquetas en ficheros .csv
fprintf('\nGuardando características en ficheros...\n')
csvwrite(strcat(DIR_GUARDAR, 'potencias_absolutas_bandas.csv'), potencias_absolutas_bandas);
csvwrite(strcat(DIR_GUARDAR, 'potencias_absolutas_total.csv'),  potencias_absolutas_total);
csvwrite(strcat(DIR_GUARDAR, 'potencias_relativas_bandas.csv'), potencias_relativas_bandas);
csvwrite(strcat(DIR_GUARDAR, 'potencias_asimetria_bandas.csv'), potencias_asimetria_bandas);
csvwrite(strcat(DIR_GUARDAR, 'potencias_asimetria_total.csv'),  potencias_asimetria_total);

csvwrite(strcat(DIR_GUARDAR, 'etiquetas_valence.csv'),   etiquetas_valence);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_arousal.csv'),   etiquetas_arousal);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_dominance.csv'), etiquetas_dominance);
csvwrite(strcat(DIR_GUARDAR, 'etiquetas_liking.csv'),    etiquetas_liking);


%% Datasets .csv preparado con características + etiquetas agrupadas en 2 clases.
% Etiquetas de 1 a 9: CLASE1: De 1 a PUNTUACION_CLASE1; CLASE2: De PUNTUACION_CLASE2 a 9.
CLASE1 = 0;
CLASE2 = 1;

ntrials_paciente   = [];      % Nº de trials por paciente.

% Valence
fv_deap = [];              % Características propuestas en el DEAP dataset: (1), (4).
fv_pt   = [];              % Características potencia total por electrodo: (2).
fv_pr   = [];              % Características potencia relativa por banda y electrodo: (3).
fv_apt  = [];              % Características asimetría potencia total: (5).
fv      = [];              % Todas las características: (1), (4), (2), (3), (5).
tv      = [];              % Etiquetas valence: CLASE1 ó CLASE2.
ntrials_paciente_v = [];

for p = 1:N_PACIENTES
    paciente = PACIENTES(p);
    c1_i     = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)<=PUNTUACION_CLASE1); 
    if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
        c2_i = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)>PUNTUACION_CLASE2);
    else 
        c2_i = (p-1)*N_TRIALS + find(etiquetas_valence(1+(p-1)*N_TRIALS:p*N_TRIALS)>=PUNTUACION_CLASE2);
    end
    fprintf("Paciente %d valence: %d (0) y %d (1)\n", paciente, length(c1_i), length(c2_i));
    f_deap = [potencias_absolutas_bandas(c1_i,:), potencias_asimetria_bandas(c1_i,:); potencias_absolutas_bandas(c2_i,:), potencias_asimetria_bandas(c2_i,:)];
    f_pt   = [potencias_absolutas_total(c1_i,:);  potencias_absolutas_total(c2_i,:)];
    f_pr   = [potencias_relativas_bandas(c1_i,:); potencias_relativas_bandas(c2_i,:)];
    f_apt  = [potencias_asimetria_total(c1_i,:);  potencias_asimetria_total(c2_i,:)];
    f_     = [f_deap, f_pt, f_pr, f_apt];
    t_     = [etiquetas_valence(c1_i)  ; etiquetas_valence(c2_i)];
    ntrials_paciente_v(p) = length(t_);
    fv_deap  = [fv_deap; f_deap];
    fv_pt    = [fv_pt; f_pt];
    fv_pr    = [fv_pr; f_pr];
    fv_apt   = [fv_apt; f_apt];
    fv       = [fv; f_];
    tv       = [tv; t_];
end
tv(tv<=PUNTUACION_CLASE1) = CLASE1; 
tv(tv>=PUNTUACION_CLASE2) = CLASE2;
dataset_v_deap    = [fv_deap tv];
dataset_v_pt      = [fv_pt tv];
dataset_v_pr      = [fv_pr tv];
dataset_v_apt     = [fv_apt tv];
dataset_v_todo    = [fv tv];
file_v_deap       = DIR_GUARDAR + "dataset1_valence_deap_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_v_pt         = DIR_GUARDAR + "dataset1_valence_pt_"   + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_v_pr         = DIR_GUARDAR + "dataset1_valence_pr_"   + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_v_apt        = DIR_GUARDAR + "dataset1_valence_apt_"  + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_v_todo       = DIR_GUARDAR + "dataset1_valence_todo_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";

csvwrite(file_v_deap, dataset_v_deap);
csvwrite(file_v_pt,   dataset_v_pt);
csvwrite(file_v_pr,   dataset_v_pr);
csvwrite(file_v_apt,  dataset_v_apt);
csvwrite(file_v_todo, dataset_v_todo);

% Arousal
fa_deap = [];              % Características propuestas en el DEAP dataset: (1), (4).
fa_pt   = [];              % Características potencia total por electrodo: (2).
fa_pr   = [];              % Características potencia relativa por banda y electrodo: (3).
fa_apt  = [];              % Características asimetría potencia total: (5).
fa      = [];              % Todas las características: (1), (4), (2), (3), (5).
ta      = [];              % Etiquetas arousal: CLASE1 ó CLASE2.
ntrials_paciente_a = [];

for p = 1:N_PACIENTES
    paciente = PACIENTES(p);
    c1_i    = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)<=PUNTUACION_CLASE1); 
    if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
        c2_i = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)>PUNTUACION_CLASE2);
    else 
        c2_i = (p-1)*N_TRIALS + find(etiquetas_arousal(1+(p-1)*N_TRIALS:p*N_TRIALS)>=PUNTUACION_CLASE2);
    end
    fprintf("Paciente %d arousal: %d (0) y %d (1)\n", paciente, length(c1_i), length(c2_i));
    f_deap = [potencias_absolutas_bandas(c1_i,:), potencias_asimetria_bandas(c1_i,:); potencias_absolutas_bandas(c2_i,:), potencias_asimetria_bandas(c2_i,:)];
    f_pt   = [potencias_absolutas_total(c1_i,:); potencias_absolutas_total(c2_i,:)];
    f_pr   = [potencias_relativas_bandas(c1_i,:); potencias_relativas_bandas(c2_i,:)];
    f_apt  = [potencias_asimetria_total(c1_i,:); potencias_asimetria_total(c2_i,:)];
    f_     = [f_deap, f_pt, f_pr, f_apt];
    t_     = [etiquetas_arousal(c1_i)  ; etiquetas_arousal(c2_i)];
    ntrials_paciente_v(p) = length(t_);
    fa_deap  = [fa_deap; f_deap];
    fa_pt    = [fa_pt; f_pt];
    fa_pr    = [fa_pr; f_pr];
    fa_apt   = [fa_apt; f_apt];
    fa       = [fa; f_];
    ta       = [ta; t_];
end
ta(ta<=PUNTUACION_CLASE1) = CLASE1; 
ta(ta>=PUNTUACION_CLASE2) = CLASE2;
dataset_a_deap    = [fa_deap ta];
dataset_a_pt      = [fa_pt ta];
dataset_a_pr      = [fa_pr ta];
dataset_a_apt     = [fa_apt ta];
dataset_a_todo    = [fa ta];
file_a_deap       = DIR_GUARDAR + "dataset1_arousal_deap_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_a_pt         = DIR_GUARDAR + "dataset1_arousal_pt_"   + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_a_pr         = DIR_GUARDAR + "dataset1_arousal_pr_"   + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_a_apt        = DIR_GUARDAR + "dataset1_arousal_apt_"  + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
file_a_todo       = DIR_GUARDAR + "dataset1_arousal_todo_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";

csvwrite(file_a_deap, dataset_a_deap);
csvwrite(file_a_pt,   dataset_a_pt);
csvwrite(file_a_pr,   dataset_a_pr);
csvwrite(file_a_apt,  dataset_a_apt);
csvwrite(file_a_todo, dataset_a_todo);

file_ntrials     = DIR_GUARDAR + "ntrials_valence_paciente_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_ntrials, ntrials_paciente_v');
file_ntrials     = DIR_GUARDAR + "ntrials_arousal_paciente_" + PUNTUACION_CLASE1 + "_" + PUNTUACION_CLASE2 + ".csv";
csvwrite(file_ntrials, ntrials_paciente_a');

