% Prueba t-Distributed Stochastic Neighbor Embedding para ver si se separan nubes de puntos

CSV_DATOS_VALENCE       = '../caracteristicas/multiple/dataset2_valence_3_7.csv';
CSV_DATOS_AROUSAL       = '../caracteristicas/multiple/dataset2_arousal_3_7.csv';
CSV_NTRIALS_PACIENTE_V  = '../caracteristicas/multiple/ntrials_valence_paciente_3_7.csv';
CSV_NTRIALS_PACIENTE_A  = '../caracteristicas/multiple/ntrials_arousal_paciente_3_7.csv';

% CSV_DATOS_AROUSAL = '../caracteristicas/dataset_arousal_5_5.csv';
% CSV_DATOS_VALENCE = '../caracteristicas/dataset_valence_5_5.csv';

PACIENTE = 4;

ntrials_paciente_v = csvread(CSV_NTRIALS_PACIENTE_V);
ntrials_paciente_a = csvread(CSV_NTRIALS_PACIENTE_A);
datos_v      = csvread(CSV_DATOS_VALENCE);
datos_a      = csvread(CSV_DATOS_AROUSAL);

% for PACIENTE = [1:2, 4, 8:32]
for PACIENTE = 1:32

    % Valence todos los pacientes
    % features_v  = datos_v(:,1:end-1);
    % tags_v      = num2cell(datos_v(:,end));
    % tags_v(datos_v(:,end)==0) = {'Low'};
    % tags_v(datos_v(:,end)==1) = {'High'};

    % Valence un paciente
    offset      = 1+sum(ntrials_paciente_v(1:(PACIENTE-1)));
    ntrials_v_p  = ntrials_paciente_v(PACIENTE);
    datos_v_p    = datos_v(offset:offset+ntrials_v_p-1,:);
    features_v   = datos_v_p(:,1:end-1);
    tags_v       = num2cell(datos_v_p(:,end));
    tags_v(datos_v_p(:,end)==0) = {'Low'};
    tags_v(datos_v_p(:,end)==1) = {'High'};


    figure; 

    rng('default') % for reproducibility
    Y = tsne(features_v,'Algorithm','exact','Distance','mahalanobis');
    subplot(2,2,1)
    gscatter(Y(:,1),Y(:,2),tags_v)
    title('Valence - Mahalanobis')

    rng('default') % for fair comparison
    Y = tsne(features_v,'Algorithm','exact','Distance','cosine');
    subplot(2,2,2)
    gscatter(Y(:,1),Y(:,2),tags_v)
    title('Valence - Cosine')

    rng('default') % for fair comparison
    Y = tsne(features_v,'Algorithm','exact','Distance','chebychev');
    subplot(2,2,3)
    gscatter(Y(:,1),Y(:,2),tags_v)
    title('Valence - Chebychev')

    rng('default') % for fair comparison
    Y = tsne(features_v,'Algorithm','exact','Distance','euclidean');
    subplot(2,2,4)
    gscatter(Y(:,1),Y(:,2),tags_v)
    title('Valence - Euclidean')


    % Arousal todos los pacientes
    % features_a  = datos_a(:,1:end-1);
    % tags_a      = num2cell(datos_a(:,end));
    % tags_a(datos_a(:,end)==0) = {'Low'};
    % tags_a(datos_a(:,end)==1) = {'High'};

    % Arousal un paciente
    offset       = 1+sum(ntrials_paciente_a(1:(PACIENTE-1)));
    ntrials_a_p  = ntrials_paciente_a(PACIENTE);
    datos_a_p    = datos_a(offset:offset+ntrials_a_p-1,:);
    features_a   = datos_a_p(:,1:end-1);
    tags_a       = num2cell(datos_a_p(:,end));
    tags_a(datos_a_p(:,end)==0) = {'Low'};
    tags_a(datos_a_p(:,end)==1) = {'High'};


    figure;

    rng('default') % for reproducibility
    Y = tsne(features_a,'Algorithm','exact','Distance','mahalanobis');
    subplot(2,2,1)
    gscatter(Y(:,1),Y(:,2),tags_a)
    title('Arousal - Mahalanobis')

    rng('default') % for fair comparison
    Y = tsne(features_a,'Algorithm','exact','Distance','cosine');
    subplot(2,2,2)
    gscatter(Y(:,1),Y(:,2),tags_a)
    title('Arousal - Cosine')

    rng('default') % for fair comparison
    Y = tsne(features_a,'Algorithm','exact','Distance','chebychev');
    subplot(2,2,3)
    gscatter(Y(:,1),Y(:,2),tags_a)
    title('Arousal - Chebychev')

    rng('default') % for fair comparison
    Y = tsne(features_a,'Algorithm','exact','Distance','euclidean');
    subplot(2,2,4)
    gscatter(Y(:,1),Y(:,2),tags_a)
    title('Arosal - Euclidean')

end