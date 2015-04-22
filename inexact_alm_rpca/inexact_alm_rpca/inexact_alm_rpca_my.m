function [A_hat E_hat iter] = inexact_alm_rpca_my(D, lambda, tol, maxIter)

% Oct 2009
% This matlab code implements the inexact augmented Lagrange multiplier 
% method for Robust PCA.
%
% D - m x n matrix of observations/data (required input)
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%     - DEFAULT 1e-7 if omitted or -1.
%
% maxIter - maximum number of iterations
%         - DEFAULT 1000, if omitted or -1.
% 
% Initialize A,E,Y,u
% while ~converged 
%   minimize (inexactly, update A and E only once)
%     L(A,E,Y,u) = |A|_* + lambda * |E|_1 + <Y,D-A-E> + mu/2 * |D-A-E|_F^2;
%   Y = Y + \mu * (D - A - E);
%   \mu = \rho * \mu;
% end
%
%
% Y - Y
% E_hat - E
% D - T
% S - S
% A_hat - T_hat
%
%
addpath PROPACK;

[m,n] = size(D);

if nargin < 2
    lambda = 1 / sqrt(m);
end

if nargin < 3
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 4
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end
%get initialization information
n = size(D, 1);
m = size(D, 2);


% initialize
Y = D;
norm_two = 1 : n;
norm_inf = 1 : n;
dual_norm = 1 : n;

for i = 1 : n
    norm_two(i) = lansvd(reshape(Y(i, : , :), m, m), 1, 'L');
    tempY = reshape(Y(i, :, :), m, m);
    norm_inf(i) = norm(tempY(:), inf)/lambda;
    dual_norm(i) = max(norm_two(i), norm_inf(i));
    Y(i, :, :) = Y(i, : , :) / dual_norm(i);
end

A_hat = zeros(1, m, m);
E_hat = zeros(n, m, m);
mu = 1.25 / sum(norm_two) * n; % this one can be tuned
mu_bar = mu * 1e7;
rho = 1.5;         % this one can be tuned
d_norm = 1 : n;
for i = 1 : n
    d_norm(i) = norm(reshape(D(i, :, :), m, m), 'fro');
end

iter = 0;
total_svd = 0;
converged = false;

sv = 10;
vecStopCriterion = 1 : n;
while ~converged       
    iter = iter + 1;
    A_hat_rep = repmat(A_hat, n, 1, 1);
    temp_T = D - A_hat_rep + (1/mu)*Y;
    E_hat = max(temp_T - lambda/mu, 0);
    E_hat = E_hat + min(temp_T + lambda/mu, 0);

    if choosvd(n, sv) == 1
        [U S V] = lansvd(reshape(sum(D - E_hat + (1/mu)*Y, 1), m, m), sv, 'L');
    else
        [U S V] = svd(reshape(sum(D - E_hat + (1/mu)*Y, 1), m, m), 'econ');
    end
    diagS = diag(S);
    svp = length(find(diagS > 1/mu));
    if svp < sv
        sv = min(svp + 1, n);
    else
        sv = min(svp + round(0.05*n), n);
    end
    
    A_hat(1, :, :) = U(:, 1:svp) * diag(diagS(1:svp) - 1/mu) * V(:, 1:svp)';    

    total_svd = total_svd + 1;
    
    A_hat_rep = repmat(A_hat, n, 1, 1);
    Z = D - A_hat_rep - E_hat;
    
    Y = Y + mu*Z;
    mu = min(mu*rho, mu_bar);
        
    %% stop Criterion   
    
for i = 1 : n
    vecStopCriterion(i) = norm(reshape(Z(i, :, :), m, m), 'fro') / d_norm(i);
end
    stopCriterion = max(vecStopCriterion);
    if stopCriterion < tol
        converged = true;
    end    
    
    if mod( total_svd, 10) == 0
        disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(reshape(A_hat, m, m)))...
            ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
            ' stopCriterion ' num2str(stopCriterion)]);
    end    
    
    if ~converged && iter >= maxIter
        disp('Maximum iterations reached') ;
        converged = 1 ;       
    end
end