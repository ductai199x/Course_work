%% Tai Duc Nguyen - ECEC 487 - 11/19/2019

clear all; close all;
seed = 0
randn('seed',seed);

M1 = [0 0];
M2 = [10 0];

S = [2 0; 0 2];
N = 500;

X1 = mvnrnd(M1, S, N);
X2 = mvnrnd(M2, S, N);

X = [X1;X2];
Y = [ones(N,1); ones(N,1)*-1];

figure
scatter(X1(:,1), X1(:,2), 'r.');
hold on
scatter(X2(:,1), X2(:,2), 'k.');
title("MY OWN KMEANS")

C_A = MY_KMEANS(X, 2);

for i=1:size(X,1)
    if C_A(i) == 1
        plot(X(i,1), X(i,2), 'go','MarkerSize',7, 'LineWidth', 1)
    else
        plot(X(i,1), X(i,2), 'ko','MarkerSize',7, 'LineWidth', 1)
    end
end
hold off


figure
scatter(X1(:,1), X1(:,2), 'r.');
hold on
scatter(X2(:,1), X2(:,2), 'k.');
title("MATLAB KMEANS")

matlab_C_A = kmeans(X, 2);

for i=1:size(X,1)
    if matlab_C_A(i) == 1
        plot(X(i,1), X(i,2), 'go','MarkerSize',7, 'LineWidth', 1)
    else
        plot(X(i,1), X(i,2), 'ko','MarkerSize',7, 'LineWidth', 1)
    end
end
hold off


function cluster_assn = MY_KMEANS(X, k)
    max_iter = 100;
    eps = 0.1;
    rand_points = randperm(size(X,1));
    k_points = zeros(k,2);
    
    cluster_assn = zeros(size(X,1),1);
    for i = 1:k
        k_points(i,:) = X(rand_points(i),:);
    end
    
    dist = zeros(1,k);
    mean_diff = 1000;
    iter = 0;
    
    while(iter < max_iter && mean_diff > eps)
        for i = 1:size(X,1)
            p = X(i,:);
            for j = 1:k
                dist(j) = sqrt(sum((p - k_points(j)).^2));
            end
            [~,idx] = min(dist);
            cluster_assn(i) = idx;
        end

        % Recompute cluster centers:
        for i = 1:k
            tmp = cluster_assn == i;
            m = mean(tmp.*X);
            mean_diff = mean_diff + sqrt(sum((m - k_points(i)).^2));
            k_points(i,:) = m;
        end
        iter = iter + 1;
    end
    

end