function [s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma] = filtrarEEG(s_eeg, Fs);    % Filtros para las diferentes bandas.

    % Filtros
    f_theta  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  4, 'PassbandFrequency2',  8, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_salpha = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  8, 'PassbandFrequency2', 10, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_alpha  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  8, 'PassbandFrequency2', 12, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_beta   = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 12, 'PassbandFrequency2', 30, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_gamma  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 30, 'PassbandFrequency2', 45, 'PassbandRipple', 1, 'SampleRate', Fs);

    % Filtrado
    s_eeg_theta     = filter(f_theta,   s_eeg);
    s_eeg_salpha    = filter(f_salpha,  s_eeg);
    s_eeg_alpha     = filter(f_alpha,   s_eeg);
    s_eeg_beta      = filter(f_beta,    s_eeg);
    s_eeg_gamma     = filter(f_gamma,   s_eeg);

end
