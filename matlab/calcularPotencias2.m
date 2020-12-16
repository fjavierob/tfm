% function [p_theta, p_theta_r, p_salpha, p_salpha_r, p_alpha, p_alpha_r, p_beta, p_beta_r, p_gamma, p_gamma_r, p_total] = calcularPotencias(n_electrodos, n_trials, pares, s_eeg_theta, s_eeg_salpha, s_eeg_alpha, s_eeg_beta, s_eeg_gamma)
function [p_theta, p_alpha, p_beta, p_total] = calcularPotencias(n_electrodos, n_trials, s_eeg_theta, s_eeg_alpha, s_eeg_beta, s_eeg_gamma)
 
    % Inicializamos las matrices donde se van a guardar las potencias
    p_theta  = zeros(1, n_trials*(n_electrodos));
    p_alpha  = zeros(1, n_trials*(n_electrodos));
    p_beta   = zeros(1, n_trials*(n_electrodos));
    p_gamma  = zeros(1, n_trials*(n_electrodos));
    p_total  = zeros(1, n_trials*(n_electrodos));
    
	% Densidades de potencia
    pd_theta   = s_eeg_theta.^2;
    pd_alpha   = s_eeg_alpha.^2;
    pd_beta    = s_eeg_beta.^2;
    pd_gamma   = s_eeg_gamma.^2;
    
    % 1) Potencias absolutas en las diferentes bandas
    p_theta(1:n_electrodos*n_trials)   = mean(pd_theta);
    p_alpha(1:n_electrodos*n_trials)   = mean(pd_alpha);
    p_beta(1:n_electrodos*n_trials)    = mean(pd_beta);
    p_gamma(1:n_electrodos*n_trials)   = mean(pd_gamma);

    % 2) Potencia total
    % p_total = p_theta + p_alpha + p_beta + p_gamma;   
    p_total = p_theta + p_alpha + p_beta;
end
