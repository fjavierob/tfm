function [s_theta, s_alpha, s_beta, s_gamma] = wavelet(s)

    waveletFunction = 'db8';

    [C,L] = wavedec(s,8,waveletFunction);

    cD1 = detcoef(C,L,1);
    cD2 = detcoef(C,L,2);
    cD3 = detcoef(C,L,3);
    cD4 = detcoef(C,L,4);
    cD5 = detcoef(C,L,5); %GAMA
    cD6 = detcoef(C,L,6); %BETA
    cD7 = detcoef(C,L,7); %ALPHA
    cD8 = detcoef(C,L,8); %THETA
    cA8 = appcoef(C,L,waveletFunction,8); %DELTA

    D1 = wrcoef('d',C,L,waveletFunction,1);
    D2 = wrcoef('d',C,L,waveletFunction,2);
    D3 = wrcoef('d',C,L,waveletFunction,3);
    D4 = wrcoef('d',C,L,waveletFunction,4);
    D5 = wrcoef('d',C,L,waveletFunction,5); %GAMMA
    D6 = wrcoef('d',C,L,waveletFunction,6); %BETA
    D7 = wrcoef('d',C,L,waveletFunction,7); %ALPHA
    D8 = wrcoef('d',C,L,waveletFunction,8); %THETA
    A8 = wrcoef('a',C,L,waveletFunction,8); %DELTA

    s_theta = detrend(D8,0);
    s_alpha = detrend(D7,0);
    s_beta  = detrend(D6,0);
    s_gamma = detrend(D5,0);

end