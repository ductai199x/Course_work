%% Tai Duc Nguyen - ECEC 487 - 10/23/2019

clear; close all;

%% Disclaimer

% Both problems 4.1 and 4.2 (below) uses the Multilayer Perceptron Neural
% Network Framework from Mo Chen (sth4nth@gmail.com) available here:
% https://www.mathworks.com/matlabcentral/fileexchange/55946-mlp-neural-network-trained-by-backpropagation
% with modifications for easy exploration of different parameters used in
% side the algorithm.

% The architecture of the neural network written by Mo 

%% Problem 4.1 Chapter 4 Page 240

x1 = [0.1 0.2 -0.15 1.1 1.2; -0.2 0.1 0.2 0.8 1.1];
x2 = [1.1 1.25 0.9 0.1 0.2; -0.1 0.15 0.1 1.2 0.9];
x = [x1 x2];

y = [ones(1,size(x1,2))*1 ones(1,size(x2,2))*-1];
lambda = 1e-3;
maxiter = 1000;

%%
p = 2;
H = 4;
m = 1;

mu = .75;
alpha = 0.001;

epoch = 4000;
MSEmin = 1e-20;

% X = [0 0 1 1;0 1 0 1];
% D = [0 1 1 0];

[Wx,Wy,MSE]=trainMLP(p,H,m,mu,alpha,x,y,epoch,MSEmin);

semilogy(MSE);

disp(['D = [' num2str(y) ']']);

t = runMLP(x,Wx,Wy);

disp(['Y = [' num2str(t) ']']);



%% 

k = [2];
actfn = @(x) tansig(x);
[model, L] = mlpReg(x',y',k,actfn,lambda,maxiter);
t = mlpRegPred(model,actfn,x');
figure;
hold on
scatter(x1(:,1),x1(:,2),'g.', 'LineWidth', 5);
scatter(x2(:,1),x2(:,2),'k.', 'LineWidth', 5);

% decision_x = linspace(min(x(:,1)), max(x(:,1)));
% for i=1:size(model.W,1)
% %     for j=1:size(model.W{i},1)
%         decision_y = -(model.W{i}(1)/model.W{i}(2))*decision_x - (model.b{i}(1)/model.W{i}(2));
%         plot(decision_x, decision_y, "r");function net = NN_training(x,y,k,code,iter,par_vec)

for i=1:size(x,1)
    if t(i) < 0
        plot(x(i,1), x(i,2), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(i,1), x(i,2), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with ', num2str(size(k,2)), ' hidden layer, each layer with ', mat2str(k), ' hidden neuron']);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'black circle = \omega_2'},'FitBoxToText','on');

%% Problem 4.1 Conclusion



%% Problem 4.2 Chapter 4 Page 240







function pe = NN_evaluation(net,x,y)
    y1 = sim(net,x); %Computation of the network outputs
    pe=sum(y.*y1<0)/length(y);
end



function net = NN_training(x,y,k,code,iter,par_vec)
    rand('seed',0) % Initialization of the random number
    % generators
    randn('seed',0) % for reproducibility of net initial
    % conditions
    % List of training methods
    methods_list = {'traingd'; 'traingdm'; 'traingda'};
    % Limits of the region where data lie
    limit = [min(x(:,1)) max(x(:,1)); min(x(:,2)) max(x(:,2))];
    % Neural network definition
    net = newff(limit,k,{'tansig','tansig'},...
    methods_list{code,1});
    % Neural network initialization
    net = init(net);
    % Setting parameters
    net.trainParam.epochs = iter;
    net.trainParam.lr=par_vec(1);
    if(code == 2)
    net.trainParam.mc=par_vec(2);
    elseif(code == 3)
    net.trainParam.lr_inc = par_vec(3);
    net.trainParam.lr_dec = par_vec(4);
    net.trainParam.max_perf_inc = par_vec(5);
    end
    % Neural network training
    net = train(net,x,y);
    %NOTE: During training, the MATLAB shows a plot of the
    % MSE vs the number of iterations.
end
