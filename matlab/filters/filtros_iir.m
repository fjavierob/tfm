function [f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_iir(Fs)
% 6 6 14 4 20
    % f_delta  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  0, 'PassbandFrequency2',  3, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_theta  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  4, 'PassbandFrequency2',  8, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_salpha = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  8, 'PassbandFrequency2', 10, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_alpha  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1',  8, 'PassbandFrequency2', 12, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_beta   = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 12, 'PassbandFrequency2', 30, 'PassbandRipple', 1, 'SampleRate', Fs);
    f_gamma  = designfilt('bandpassiir', 'FilterOrder', 20, 'PassbandFrequency1', 30, 'PassbandFrequency2', 45, 'PassbandRipple', 1, 'SampleRate', Fs);
end