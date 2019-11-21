%% Tai Duc Nguyen - ECEC 487 - 11/15/2019

clear all; close all;
seed = 0
randn('seed',seed);

%% How close can the two distributions get before the gap statistic fails to differentiate them?

M1 = [0 0];
M2 = [2.3 0];

S = [1 0; 0 1];
N = 500;

X1 = mvnrnd(M1, S, N);
X2 = mvnrnd(M2, S, N);

X = [X1;X2];
Y = [ones(N,1); ones(N,1)*2];

figure
scatter(X1(:,1), X1(:,2), 'g.');
hold on
scatter(X2(:,1), X2(:,2), 'k.');

E = evalclusters(X, 'kmeans', 'gap', 'KList', [1:3], 'Distance', 'sqEuclidean');

for i=1:size(X,1)
    if E.OptimalY(i) == 1
        plot(X(i,1), X(i,2), 'ko','MarkerSize',7, 'LineWidth', 1)
    else
        plot(X(i,1), X(i,2), 'go','MarkerSize',7, 'LineWidth', 1)
    end
end
hold off

figure
plot(E)