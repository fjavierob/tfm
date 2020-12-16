%% 4.1. Extracción características

% Filtros
Fs = 128;  % Hz.
[f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_iir(Fs);
% fvtool(f_theta); 
% fvtool(f_salpha);
% fvtool(f_alpha);
% fvtool(f_beta);
% fvtool(f_gamma);

% Filtrar señal
DIR_DATASET     = 'C:\Users\Javi\Documents\TFM\Dataset';
paciente        = 1;
fileName        = sprintf('s%02d.mat', paciente);
datos_paciente  = fullfile(DIR_DATASET, fileName);
load(datos_paciente);
eeg             = data(1,1,:);
eeg             = eeg(:);
eeg_beta        = filter(f_beta, eeg);
L               = length(eeg);
t  = 0:L-1;  t  = t/Fs;
figure;
subplot(2,1,1);   plot(t(1:10*Fs), eeg(1:10*Fs));
title('EEG')
xlabel('t (s)');
subplot(2,1,2);   plot(t(1:10*Fs), eeg_beta(1:10*Fs));
title('EEG en banda beta (12 - 30 Hz)')
xlabel('t (s)'); 
f = 0:Fs/L:Fs/2;
f_eeg           = fft(eeg);
f_eeg           = f_eeg(1:L/2+1);
f_eeg_beta      = fft(eeg_beta);
f_eeg_beta      = f_eeg_beta(1:L/2+1);
figure;
subplot(2,1,1);   plot(f, f_eeg);
title('EEG')
xlabel('f (Hz)');
subplot(2,1,2);   plot(f, f_eeg_beta);
title('EEG en banda beta (12 - 30 Hz)')
xlabel('f (Hz)');