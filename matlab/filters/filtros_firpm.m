function [f_theta, f_salpha, f_alpha, f_beta, f_gamma] = filtros_firpm(Fs)

    %% Theta
    Fstop1 = 2.5;             % First Stopband Frequency
    Fpass1 = 3;               % First Passband Frequency
    Fpass2 = 7;               % Second Passband Frequency
    Fstop2 = 7.5;             % Second Stopband Frequency
    Dstop1 = 0.001;           % First Stopband Attenuation
    Dpass  = 0.057501127785;  % Passband Ripple
    Dstop2 = 0.0001;          % Second Stopband Attenuation
    dens   = 20;              % Density Factor
    % Calculate the order from the parameters using FIRPMORD.
    [N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                            0], [Dstop1 Dpass Dstop2]);
    % Calculate the coefficients using the FIRPM function.
    b  = firpm(N, Fo, Ao, W, {dens});
    f_theta = dfilt.dffir(b);


    %% Alpha
    Fstop1 = 7.5;             % First Stopband Frequency
    Fpass1 = 8;               % First Passband Frequency
    Fpass2 = 10;              % Second Passband Frequency
    Fstop2 = 10.5;            % Second Stopband Frequency
    Dstop1 = 0.0001;          % First Stopband Attenuation
    Dpass  = 0.057501127785;  % Passband Ripple
    Dstop2 = 0.0001;          % Second Stopband Attenuation
    dens   = 20;              % Density Factor
    % Calculate the order from the parameters using FIRPMORD.
    [N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                            0], [Dstop1 Dpass Dstop2]);
    % Calculate the coefficients using the FIRPM function.
    b  = firpm(N, Fo, Ao, W, {dens});
    f_salpha = dfilt.dffir(b);


    %% Alpha
    Fstop1 = 7.5;             % First Stopband Frequency
    Fpass1 = 8;               % First Passband Frequency
    Fpass2 = 12;              % Second Passband Frequency
    Fstop2 = 12.5;            % Second Stopband Frequency
    Dstop1 = 0.0001;          % First Stopband Attenuation
    Dpass  = 0.057501127785;  % Passband Ripple
    Dstop2 = 0.0001;          % Second Stopband Attenuation
    dens   = 20;              % Density Factor
    % Calculate the order from the parameters using FIRPMORD.
    [N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                            0], [Dstop1 Dpass Dstop2]);
    % Calculate the coefficients using the FIRPM function.
    b  = firpm(N, Fo, Ao, W, {dens});
    f_alpha = dfilt.dffir(b);


    %% Beta
    Fstop1 = 11.5;            % First Stopband Frequency
    Fpass1 = 12;              % First Passband Frequency
    Fpass2 = 30;              % Second Passband Frequency
    Fstop2 = 30.5;            % Second Stopband Frequency
    Dstop1 = 0.0001;          % First Stopband Attenuation
    Dpass  = 0.057501127785;  % Passband Ripple
    Dstop2 = 0.0001;          % Second Stopband Attenuation
    dens   = 20;              % Density Factor
    % Calculate the order from the parameters using FIRPMORD.
    [N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                            0], [Dstop1 Dpass Dstop2]);
    % Calculate the coefficients using the FIRPM function.
    b  = firpm(N, Fo, Ao, W, {dens});
    f_beta = dfilt.dffir(b);


    %% Gamma
    Fstop1 = 19.5;            % First Stopband Frequency
    Fpass1 = 30;              % First Passband Frequency
    Fpass2 = 47;              % Second Passband Frequency
    Fstop2 = 47.5;            % Second Stopband Frequency
    Dstop1 = 0.0001;          % First Stopband Attenuation
    Dpass  = 0.057501127785;  % Passband Ripple
    Dstop2 = 0.0001;          % Second Stopband Attenuation
    dens   = 20;              % Density Factor
    % Calculate the order from the parameters using FIRPMORD.
    [N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                            0], [Dstop1 Dpass Dstop2]);
    % Calculate the coefficients using the FIRPM function.
    b  = firpm(N, Fo, Ao, W, {dens});
    f_gamma = dfilt.dffir(b);

end


