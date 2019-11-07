%% Tai Duc Nguyen - ECEC 487 - 10/23/2019

clear; close all;

%% Disclaimer

% Both problems 4.1 and 4.2 (below) uses the Multilayer Perceptron Neural
% Network Framework from Marcelo Augusto Costa Fernandes (mfernandes@dca.ufrn.br) available here:
% https://www.mathworks.com/matlabcentral/fileexchange/36253-the-matrix-implementation-of-the-two-layer-multilayer-perceptron-mlp-neural-networks
% with modifications for easy exploration of different parameters used in
% side the algorithm.

% The architecture of the neural network written by Fernandes is a simple 1
% input layer size p, 1 hidden layer with number of neurons H, and 1 output
% layer size m. The error calculation uses Mean Square Error. This
% algoirthm only back-propagates after finishing feed-forwarding on a batch
% of inputs.

%% Problem 4.1 Chapter 4 Page 240

x1 = [0.1 0.2 -0.15 1.1 1.2; -0.2 0.1 0.2 0.8 1.1];
x2 = [1.1 1.25 0.9 0.1 0.2; -0.1 0.15 0.1 1.2 0.9];
x = [x1 x2];

y = [ones(1,size(x1,2))*1 ones(1,size(x2,2))*0];

%% H = 1; maxiter = 5000;
p = 2;
H = 1;
m = 1;

maxiter = 5000;
mu = .5;
alpha = 0;
MSEmin = 1e-20;
actfn1 = @(x) (1./(1+exp(-x)));
actfn2 = @(x) (1./(1+exp(-x)));

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,actfn1,maxiter,MSEmin);

figure();
semilogy(MSE);
title(['MSE for network with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
xlabel('epoch'); ylabel('MSE');

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy,actfn2);

disp(['Y = [' num2str(t) ']']);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter), 'mu=', num2str(mu)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');

%% H = 1; maxiter = 50000;
p = 2;
H = 1;
m = 1;

maxiter = 50000;
mu = .5;
alpha = 0;
MSEmin = 1e-20;
actfn1 = @(x) (1./(1+exp(-x)));
actfn2 = @(x) (1./(1+exp(-x)));

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,actfn1,maxiter,MSEmin);

figure();
semilogy(MSE);
title(['MSE for network with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
xlabel('epoch'); ylabel('MSE');

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy,actfn2);

disp(['Y = [' num2str(t) ']']);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter), 'mu=', num2str(mu)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');


%% H = 3; maxiter = 10000;
p = 2;
H = 3;
m = 1;

maxiter = 10000;
mu = .75;
alpha = 0;
MSEmin = 1e-20;
actfn1 = @(x) (1./(1+exp(-x)));
actfn2 = @(x) (1./(1+exp(-x)));

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,actfn1,maxiter,MSEmin);

figure();
semilogy(MSE);
title(['MSE for network with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
xlabel('epoch'); ylabel('MSE');

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy,actfn2);

disp(['Y = [' num2str(t) ']']);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');


%% Problem 4.2

M1 = [0 0];
M2 = [1 1];
M3 = [0 1];
M4 = [1 0];

S = [0.01 0; 0 0.01];
N = 100;

a1 = mvnrnd(M1, S, N);
a2 = mvnrnd(M2, S, N);
a3 = mvnrnd(M3, S, N);
a4 = mvnrnd(M4, S, N);

x1 = [a1' a2'];
x2 = [a3' a4'];

x = [x1 x2];

y = [ones(1,N*2) ones(1,N*2)*0];

%% maxiter = 10000; mu (learning param) = 0.2
p = 2;
H = 2;
m = 1;

maxiter = 10000;
mu = .2;
alpha = 0;
MSEmin = 1e-20;
actfn1 = @(x) (1./(1+exp(-x)));
actfn2 = @(x) (1./(1+exp(-x)));

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,actfn1,maxiter,MSEmin);

figure();
semilogy(MSE);
title(['MSE for network with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
xlabel('epoch'); ylabel('MSE');

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy,actfn2);

disp(['Y = [' num2str(t) ']']);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter), 'mu=', num2str(mu)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');

%% maxiter = 10000; mu (learning param) = 0.75
p = 2;
H = 2;
m = 1;

maxiter = 10000;
mu = .75;
alpha = 0;
MSEmin = 1e-20;
actfn1 = @(x) (1./(1+exp(-x)));
actfn2 = @(x) (1./(1+exp(-x)));

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,actfn1,maxiter,MSEmin);

figure();
semilogy(MSE);
title(['MSE for network with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter)]);
xlabel('epoch'); ylabel('MSE');

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy,actfn2);

disp(['Y = [' num2str(t) ']']);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter), 'mu=', num2str(mu)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');


%% Produce 50 more vectors, use calculated weights to classify.

M1 = [0 0];
M2 = [1 1];
M3 = [0 1];
M4 = [1 0];

S = [0.01 0; 0 0.01];
N = 50;

a1 = mvnrnd(M1, S, N);
a2 = mvnrnd(M2, S, N);
a3 = mvnrnd(M3, S, N);
a4 = mvnrnd(M4, S, N);

x1 = [a1' a2'];
x2 = [a3' a4'];

x = [x1 x2];

y = [ones(1,N*2) ones(1,N*2)*0];

t = runMLP(x,Wx,Wy,actfn2);

figure();
hold on
scatter(x1(1,:),x1(2,:),'g.', 'LineWidth', 5);
scatter(x2(1,:),x2(2,:),'k.', 'LineWidth', 5);

for i=1:size(x,2)
    if t(i) > 0.5
        plot(x(1,i), x(2,i), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(1,i), x(2,i), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with 1 hidden layer with ', num2str(H), ' hidden neurons; max iter=', num2str(maxiter), 'mu=', num2str(mu)]);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');

