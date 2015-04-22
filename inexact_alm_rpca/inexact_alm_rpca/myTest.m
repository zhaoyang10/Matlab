clear
clc;

path = 'G:\code\MATLAB\RobustLateFusionWithRankMinimization\myRobust\';
addpath(genpath(path));
load([path,'flower17\distancematrices17gcfeat06.mat']);
load([path,'flower17\distancematrices17itfeat08.mat']);
load([path,'flower17\datasplits.mat']);
D(:,:,1) = D_colourgc;
D(:,:,2) = D_hog;
D(:,:,3) = D_hsv;
D(:,:,4) = D_shapegc;
D(:,:,5) = D_siftbdy;
D(:,:,6) = D_siftint;
D(:,:,7) = D_texturegc;
numsam = size(D,1);
numker = size(D,3);
numclass = 17;
Y = zeros(1360,1);
for ic =1:numclass
    Y(80*(ic-1)+1:80*ic)=ic;
end
K = zeros(numsam,numsam,numker);
for p =1:7
    sigmap = mean(mean(D(:,:,p)));
    K(:,:,p) = exp(-D(:,:,p)/sigmap);
end
%% K = kcenter(K); K = knorm(K);
Ktrntrn = K(trn1,trn1,:); Kvaltrn = K(val1,trn1,:); Ktsttrn = K(tst1,trn1,:);
Ytrn = Y(trn1); Yval = Y(val1); Ytst = Y(tst1);

optimalC1 = 10;
Sigma1 = 1;
numtrn = size(Ktrntrn,1);
KUtrntrn = zeros(numtrn,numtrn,numker);
for p =1:numker
    [U] = myspcl(Ktrntrn(:,:,p), numclass);
    KUtrntrn(:,:,p) = U*U';
end
[A_hat,E_hat,iter] = inexact_alm_multi_rpca(Ktrntrn);
[U1] = myspcl(A_hat, numclass);
%{
[acc1(1),acc1(2)]= accuFuc(U1,Ytrn,numclass)
avgKer = combFun(Ktrntrn);
[U2] = myspcl(avgKer,numclass);
[acc2(1),acc2(2)]= accuFuc(U2,Ytrn,numclass)
for p =1:numker
    [Up] = myspcl(Ktrntrn(:,:,p),numclass);
    [acc(p,1),acc(p,2)]= accuFuc(Up,Ytrn,numclass);
end
acc3=max(acc)
%}