% Script para probar el clasificador LDA

warning('off');

PUNTUACION_CLASE1   = 3;
PUNTUACION_CLASE2   = 7;
PACIENTES           = 1:32;
BUENA_CLASIFICACION = 0.6;
N_PACIENTES         = length(PACIENTES);
TIPO                = ["valence", "arousal"];

% Extraer características paper 
% caracteristicas2

% Clasificar pacientes de forma individual
acierto_pacientes_v = zeros(N_PACIENTES, 1);
for paciente = PACIENTES
    acierto_pacientes_v(paciente) = clasificador_lda(paciente, "valence", PUNTUACION_CLASE1, PUNTUACION_CLASE2);
end
acierto_pacientes_a = zeros(N_PACIENTES, 1);
for paciente = PACIENTES
    acierto_pacientes_a(paciente) = clasificador_lda(paciente, "arousal",  PUNTUACION_CLASE1, PUNTUACION_CLASE2);
end

% Clasificar con los datos de todos los pacientes a la vez
acierto_v = clasificador_lda(0, "valence", PUNTUACION_CLASE1, PUNTUACION_CLASE2);
acierto_a = clasificador_lda(0, "arousal", PUNTUACION_CLASE1, PUNTUACION_CLASE2);

% Resultados
p_buenos_v = find(acierto_pacientes_v>=BUENA_CLASIFICACION);
p_buenos_a = find(acierto_pacientes_a>=BUENA_CLASIFICACION);
fprintf('\n');
fprintf('        Clasificación valence         \t\t        Clasificación arousal         \n');
fprintf('--------------------------------------\t\t--------------------------------------\n');
fprintf('Pacientes juntos:\t\t%.2f%%          Pacientes juntos:\t\t%.2f%%           \n',        100*acierto_v, 100*acierto_a);
fprintf('Media pacientes separados:\t%.2f%%    \tMedia pacientes separados:\t%.2f%%    \n',     100*mean(acierto_pacientes_v), ...
                                                                                                100*mean(acierto_pacientes_a));
fprintf('%d pacientes >= %.2f%%                \t\t%d pacientes >= %.2f%%                \n',   length(p_buenos_v), 100*BUENA_CLASIFICACION, ...
                                                                                                length(p_buenos_a), 100*BUENA_CLASIFICACION);

% for p = p_buenos_v.'
%     fprintf('  Valence #%02d:\t%2.2f%%\n', p, 100*acierto_pacientes_v(p));
% end
% fprintf("\n")
% for p = p_buenos_a.'
%     fprintf('  Arousal #%02d:\t%2.2f%%\n', p, 100*acierto_pacientes_a(p));
% end


