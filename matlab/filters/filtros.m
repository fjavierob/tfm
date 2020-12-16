Fs = 128;
t = 0:1/Fs:6-1/Fs;

% Señal de prueba
s_theta  = cos(2*pi*4*t) + 2*cos(2*pi*5*t);
s_salpha = 2*cos(2*pi*9*t); 
s_alpha  = s_salpha + 1*cos(2*pi*11*t);  
s_beta   = cos(2*pi*16*t) + 2*cos(2*pi*22*t) + 3*cos(2*pi*28*t);
s_gamma  = cos(2*pi*32*t) + cos(2*pi*40*t); 

s_eeg = s_theta + s_alpha + s_beta + s_gamma;

% Filtros IIR
[f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_iir(Fs);

% Filtrado de la señal con los diferentes filtros
s_theta_  = filter(f_theta,  s_eeg);
s_salpha_ = filter(f_salpha, s_eeg);
s_alpha_  = filter(f_alpha,  s_eeg);
s_beta_   = filter(f_beta,   s_eeg);
s_gamma_  = filter(f_gamma,  s_eeg);

% Filtros FIRPM
% [f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_iir2(Fs);

% s_theta__  = filter(f_theta,  s_eeg);
% s_salpha__ = filter(f_salpha, s_eeg);
% s_alpha__  = filter(f_alpha,  s_eeg);
% s_beta__   = filter(f_beta,   s_eeg);
% s_gamma__  = filter(f_gamma,  s_eeg);

% Wavelet decomposition
% [s_theta__, s_alpha__, s_beta__, s_gamma__] = wavelet(s_eeg);
% s_salpha__ = s_salpha_;

s_theta__  = filtfilt(f_theta,  s_eeg);
s_salpha__ = filtfilt(f_salpha, s_eeg);
s_alpha__  = filtfilt(f_alpha,  s_eeg);
s_beta__   = filtfilt(f_beta,   s_eeg);
s_gamma__  = filtfilt(f_gamma,  s_eeg);

% Comparamos las señales originales de cada banda con las obtenidas tras filtrar
% tiempo
figure; 

subplot(5,3,1);  plot(t, s_theta);
subplot(5,3,4);  plot(t, s_salpha);
subplot(5,3,7);  plot(t, s_alpha);
subplot(5,3,10); plot(t, s_beta);
subplot(5,3,13); plot(t, s_gamma);

subplot(5,3,2);  plot(t, s_theta_);
subplot(5,3,5);  plot(t, s_salpha_);
subplot(5,3,8);  plot(t, s_alpha_);
subplot(5,3,11); plot(t, s_beta_);
subplot(5,3,14); plot(t, s_gamma_);

subplot(5,3,3);  plot(t, s_theta__);a=axis;
subplot(5,3,6);  plot(t, s_salpha__);axis(a)
subplot(5,3,9);  plot(t, s_alpha__);axis(a)
subplot(5,3,12); plot(t, s_beta__);axis(a)
subplot(5,3,15); plot(t, s_gamma__);axis(a)

% frecuencia
L = length(t);
f = 0:Fs/L:Fs/2;

sf_theta  = fft(s_theta);   sf_theta  = sf_theta(1:L/2+1);
sf_salpha = fft(s_salpha);  sf_salpha = sf_salpha(1:L/2+1);
sf_alpha  = fft(s_alpha);   sf_alpha  = sf_alpha(1:L/2+1);
sf_beta   = fft(s_beta);    sf_beta   = sf_beta(1:L/2+1);
sf_gamma  = fft(s_gamma);   sf_gamma  = sf_gamma(1:L/2+1);

sf_theta_  = fft(s_theta_);     sf_theta_  = sf_theta_(1:L/2+1);
sf_salpha_ = fft(s_salpha_);    sf_salpha_ = sf_salpha_(1:L/2+1);
sf_alpha_  = fft(s_alpha_);     sf_alpha_  = sf_alpha_(1:L/2+1);
sf_beta_   = fft(s_beta_);      sf_beta_   = sf_beta_(1:L/2+1);
sf_gamma_  = fft(s_gamma_);     sf_gamma_  = sf_gamma_(1:L/2+1);

sf_theta__  = fft(s_theta__);   sf_theta__  = sf_theta__(1:L/2+1);
sf_salpha__ = fft(s_salpha__);  sf_salpha__ = sf_salpha__(1:L/2+1);
sf_alpha__  = fft(s_alpha__);   sf_alpha__  = sf_alpha__(1:L/2+1);
sf_beta__   = fft(s_beta__);    sf_beta__   = sf_beta__(1:L/2+1);
sf_gamma__  = fft(s_gamma__);   sf_gamma__  = sf_gamma__(1:L/2+1);

figure;

subplot(5,3,1);  plot(f, abs(sf_theta));
subplot(5,3,4);  plot(f, abs(sf_salpha));
subplot(5,3,7);  plot(f, abs(sf_alpha));
subplot(5,3,10); plot(f, abs(sf_beta));
subplot(5,3,13); plot(f, abs(sf_gamma));

subplot(5,3,2);  plot(f, abs(sf_theta_));
subplot(5,3,5);  plot(f, abs(sf_salpha_));
subplot(5,3,8);  plot(f, abs(sf_alpha_));
subplot(5,3,11); plot(f, abs(sf_beta_));
subplot(5,3,14); plot(f, abs(sf_gamma_));

subplot(5,3,3);  plot(f, abs(sf_theta__));
subplot(5,3,6);  plot(f, abs(sf_salpha__));
subplot(5,3,9);  plot(f, abs(sf_alpha__));
subplot(5,3,12); plot(f, abs(sf_beta__));
subplot(5,3,15); plot(f, abs(sf_gamma__));

% Comparamos las PSD
psd_sf_theta   = (1/(Fs*L)) * abs(sf_theta).^2;         psd_sf_theta(2:end-1)   = 2*psd_sf_theta(2:end-1);
psd_sf_salpha  = (1/(Fs*L)) * abs(sf_salpha).^2;        psd_sf_salpha(2:end-1)  = 2*psd_sf_salpha(2:end-1);
psd_sf_alpha   = (1/(Fs*L)) * abs(sf_alpha).^2;         psd_sf_alpha(2:end-1)   = 2*psd_sf_alpha(2:end-1);
psd_sf_beta    = (1/(Fs*L)) * abs(sf_beta).^2;          psd_sf_beta(2:end-1)    = 2*psd_sf_beta(2:end-1);
psd_sf_gamma   = (1/(Fs*L)) * abs(sf_gamma).^2;         psd_sf_gamma(2:end-1)   = 2*psd_sf_gamma(2:end-1);

psd_sf_theta_   = (1/(Fs*L)) * abs(sf_theta_).^2;       psd_sf_theta_(2:end-1)   = 2*psd_sf_theta_(2:end-1);
psd_sf_salpha_  = (1/(Fs*L)) * abs(sf_salpha_).^2;      psd_sf_salpha_(2:end-1)  = 2*psd_sf_salpha_(2:end-1);
psd_sf_alpha_   = (1/(Fs*L)) * abs(sf_alpha_).^2;       psd_sf_alpha_(2:end-1)   = 2*psd_sf_alpha_(2:end-1);
psd_sf_beta_    = (1/(Fs*L)) * abs(sf_beta_).^2;        psd_sf_beta_(2:end-1)    = 2*psd_sf_beta_(2:end-1);
psd_sf_gamma_   = (1/(Fs*L)) * abs(sf_gamma_).^2;       psd_sf_gamma_(2:end-1)   = 2*psd_sf_gamma_(2:end-1);

psd_sf_theta__   = (1/(Fs*L)) * abs(sf_theta__).^2;     psd_sf_theta__(2:end-1)   = 2*psd_sf_theta__(2:end-1);
psd_sf_salpha__  = (1/(Fs*L)) * abs(sf_salpha__).^2;    psd_sf_salpha__(2:end-1)  = 2*psd_sf_salpha__(2:end-1);
psd_sf_alpha__   = (1/(Fs*L)) * abs(sf_alpha__).^2;     psd_sf_alpha__(2:end-1)   = 2*psd_sf_alpha__(2:end-1);
psd_sf_beta__    = (1/(Fs*L)) * abs(sf_beta__).^2;      psd_sf_beta__(2:end-1)    = 2*psd_sf_beta__(2:end-1);
psd_sf_gamma__   = (1/(Fs*L)) * abs(sf_gamma__).^2;     psd_sf_gamma__(2:end-1)   = 2*psd_sf_gamma__(2:end-1);

figure;

subplot(5,3,1);  plot(f, 10*log10(psd_sf_theta));
subplot(5,3,4);  plot(f, 10*log10(psd_sf_salpha));
subplot(5,3,7);  plot(f, 10*log10(psd_sf_alpha));
subplot(5,3,10); plot(f, 10*log10(psd_sf_beta));
subplot(5,3,13); plot(f, 10*log10(psd_sf_gamma));

subplot(5,3,2);  plot(f, 10*log10(psd_sf_theta_));
subplot(5,3,5);  plot(f, 10*log10(psd_sf_salpha_));
subplot(5,3,8);  plot(f, 10*log10(psd_sf_alpha_));
subplot(5,3,11); plot(f, 10*log10(psd_sf_beta_));
subplot(5,3,14); plot(f, 10*log10(psd_sf_gamma_));

subplot(5,3,3);  plot(f, 10*log10(psd_sf_theta__));
subplot(5,3,6);  plot(f, 10*log10(psd_sf_salpha__));
subplot(5,3,9);  plot(f, 10*log10(psd_sf_alpha__));
subplot(5,3,12); plot(f, 10*log10(psd_sf_beta__));
subplot(5,3,15); plot(f, 10*log10(psd_sf_gamma__));

% Comparamos las potencias
% tiempo
fprintf("\t\tPotencias tiempo\n");
fprintf("--------------------------------------------------------\n");
fprintf("Banda\t\tOriginal\tfiltro1\t\tfiltro2\n");
fprintf("-----\t\t--------\t------\t\t--------\n")

p_theta   = mean(s_theta.^2);
p_theta_  = mean(s_theta_.^2);
p_theta__ = mean(s_theta__.^2);
fprintf("theta\t\t%.3f\t\t%.3f\t\t%.3f\n", p_theta, p_theta_, p_theta__);

p_salpha   = mean(s_salpha.^2);
p_salpha_  = mean(s_salpha_.^2);
p_salpha__ = mean(s_salpha__.^2);
fprintf("slow alpha\t%.3f\t\t%.3f\t\t%.3f\n", p_salpha, p_salpha_, p_salpha__);

p_alpha   = mean(s_alpha.^2);
p_alpha_  = mean(s_alpha_.^2);
p_alpha__ = mean(s_alpha__.^2);
fprintf("alpha\t\t%.3f\t\t%.3f\t\t%.3f\n", p_alpha, p_alpha_, p_alpha__);

p_beta   = mean(s_beta.^2);
p_beta_  = mean(s_beta_.^2);
p_beta__ = mean(s_beta__.^2);
fprintf("beta\t\t%.3f\t\t%.3f\t\t%.3f\n", p_beta, p_beta_, p_beta__);

p_gamma   = mean(s_gamma.^2);
p_gamma_  = mean(s_gamma_.^2);
p_gamma__ = mean(s_gamma__.^2);
fprintf("gamma\t\t%.3f\t\t%.3f\t\t%.3f\n", p_gamma, p_gamma_, p_gamma__);

% frecuencia
fprintf("\n\t\tPotencias frecuencia\n");
fprintf("--------------------------------------------------------\n");
fprintf("Banda\t\tOriginal\tfiltro1\t\tfiltro2\n");
fprintf("-----\t\t--------\t------\t\t--------\n")

p_theta   = mean(abs(sf_theta.^2))/L;
p_theta_  = mean(abs(sf_theta_.^2))/L;
p_theta__ = mean(abs(sf_theta__.^2));
fprintf("theta\t\t%.3f\t\t%.3f\t\t%.3f\n", p_theta, p_theta_, p_theta__);

p_salpha   = mean(abs(sf_salpha.^2))/L;
p_salpha_  = mean(abs(sf_salpha_.^2))/L;
p_salpha__ = mean(abs(sf_salpha__.^2));
fprintf("slow alpha\t%.3f\t\t%.3f\t\t%.3f\n", p_salpha, p_salpha_, p_salpha__);

p_alpha   = mean(abs(sf_alpha.^2))/L;
p_alpha_  = mean(abs(sf_alpha_.^2))/L;
p_alpha__ = mean(abs(sf_alpha__.^2));
fprintf("alpha\t\t%.3f\t\t%.3f\t\t%.3f\n", p_alpha, p_alpha_, p_alpha__);

p_beta   = mean(abs(sf_beta.^2))/L;
p_beta_  = mean(abs(sf_beta_.^2))/L;
p_beta__ = mean(abs(sf_beta__.^2));
fprintf("beta\t\t%.3f\t\t%.3f\t\t%.3f\n", p_beta, p_beta_, p_beta__);

p_gamma   = mean(abs(sf_gamma.^2))/L;
p_gamma_  = mean(abs(sf_gamma_.^2))/L;
p_gamma__ = mean(abs(sf_gamma__.^2));
fprintf("gamma\t\t%.3f\t\t%.3f\t\t%.3f\n", p_gamma, p_gamma_, p_gamma__);

