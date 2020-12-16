% Función para entrenar y probar un clasificador LDA, siguiendo el pseudocódigo del paper de Sergio, Rubén y Auxi.
%
% Parámetros de entrada:
%
%  - PACIENTE:  nº de paciente para el que queremos clasificar (1-32).
%               Si queremos clasificar para todos en conjunto: Indicar un número fuera del rango 1-32.
%
%  - tipo:      "valence" ó "arousal" según queramos clasificar valencia o arousal.
%
%  - PUNTUACION_CLASE1:  Queremos realizar una clasificación binaria, pero nuestras etiquetas son ratings
%  - PUNTUACION_CLASE2:  que van de 1 a 9. Agrupamos en dos clases: CLASE1: Ratings de 1 a PUNTUACION_CLASE1,
%                        CLASE2: Ratings de PUNTUACION_CLASE2 a 9.
%  
% Salida:
% 
%  - acierto: tasa de acierto del clasificador en tanto por 1.
function acierto = clasificador1(PACIENTE, tipo, PUNTUACION_CLASE1, PUNTUACION_CLASE2)

N_TRIALS = 40;
CLASE1   = 0;
CLASE2   = 1;
TEST     = 0.3;

%% Cargar características y etiquetas
switch tipo
    case "valence"
        f = csvread('caracteristicas_valence.csv');
        t = csvread('./etiquetas_valence.csv');
    case "arousal"
        f = csvread('caracteristicas_arousal.csv');
        t = csvread('./etiquetas_arousal.csv');
end

% Me quedo con los experimentos para un paciente determinado si PACIENTE en intervalo [1,32]
if (PACIENTE >= 1 && PACIENTE <= 32)
    f = f(1+(PACIENTE-1)*N_TRIALS:PACIENTE*N_TRIALS,:);
    t = t(1+(PACIENTE-1)*N_TRIALS:PACIENTE*N_TRIALS,:);
end

%% Preparación datos
% Agrupar etiquetas en dos clases
c1_i     = find(t<=PUNTUACION_CLASE1); 
if (PUNTUACION_CLASE1 == PUNTUACION_CLASE2)
    c2_i = find(t>PUNTUACION_CLASE2);
else 
    c2_i = find(t>=PUNTUACION_CLASE2);
end
f       = [f(c1_i,:) ; f(c2_i,:)];
t       = [t(c1_i)   ; t(c2_i)];
t(t<=PUNTUACION_CLASE1) = CLASE1; 
t(t>=PUNTUACION_CLASE2) = CLASE2;

% Divido en sets de entrenamiento y test
c1_i  = find(t==CLASE1);    f1 = f(c1_i,:);     t1 = t(c1_i,:);  l1 = size(f1,1);
c2_i  = find(t==CLASE2);    f2 = f(c2_i,:);     t2 = t(c2_i,:);  l2 = size(f2,1);

n1_train = round((1-TEST)*l1);  n1_test = l1-n1_train;
n2_train = round((1-TEST)*l2);  n2_test = l2-n2_train;

f = [f1(1:n1_train,:) ; f2(1:n2_train,:)];
t = [t1(1:n1_train,:) ; t2(1:n2_train,:)];

f_test  = [f1(1+n1_train:end,:) ; f2(1+n2_train:end,:)].';
t_test  = [t1(1+n1_train:end,:) ; t2(1+n2_train:end,:)].';

%% Entrenar el clasificador LDA

N = size(f,1);
% Clase 1
c1_i        = find(t==CLASE1);
f1          = f(c1_i,:).';
N1          = size(f1, 1);
u1          = mean(f1, 2);       
s1          = cov(f1.');
p1          = N1/N;

% Clase 2
c2_i        = find(t==CLASE2);
f2          = f(c2_i,:).';
N2          = size(f2, 1);
u2          = mean(f2, 2);      
s2          = cov(f2.');
p2          = N2/N;

s           = p1*s1 + p2*s2;
x           = inv(s)*(u1-u2);
b           = 0.5*(u1+u2) - (u1-u2)*(log(p1)-log(p2))./((u1-u2).'*x);  


%% Validación

% Prueba datos test
k = 0.5+0.5*sign(x.'*(f_test-b));

acierto = length(find(k==t_test))/length(k);

% fprintf("Porcentaje acierto: %.2f%%\n", acierto);
end

