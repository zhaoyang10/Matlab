function [U] = myspcl(S,numclus)

%%all parameters are got%%%
num = size(S,1);
W = diag(sum(S,2));

L = eye(num) - (W^-0.5) * S * (W^-0.5);
L = (L+L')/2;
[evectors, evalues] = eigs(L,numclus+1,'SM');
U = evectors(:,2:end);
%% normalization
U = bsxfun(@rdivide,U,sqrt(sum(U.^2,2)));
