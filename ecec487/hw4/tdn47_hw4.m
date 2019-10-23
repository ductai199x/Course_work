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

x1 = [0.1 0.2 -0.15 1.1 1.2; -0.2 0.1 0.2 0.8 1.1]';
x2 = [1.1 1.25 0.9 0.1 0.2; -0.1 0.15 0.1 1.2 0.9]';
x = [x1;x2];

y = [ones(size(x1,1),1)*1; ones(size(x2,1),1)*-1];
lambda = 1e-3;
maxiter = 50000;
%% k = []

k = [3 3];
actfn = @(x) heaviside(x)*2-1;
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
%         plot(decision_x, decision_y, "r");
% %     end
% end

for i=1:size(x,1)
    if t(i) > 0
        plot(x(i,1), x(i,2), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(i,1), x(i,2), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with ', num2str(size(k,2)), ' hidden layer, each layer with ', mat2str(k), ' hidden neuron']);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'red circle = \omega_2'},'FitBoxToText','on');

%% k = [2]

k = [2];
actfn = @(x) heaviside(x)*2-1;
[model, L] = mlpReg(x',y',k,actfn,lambda,maxiter);
t = mlpRegPred(model,actfn,x');
figure;
hold on
scatter(x1(:,1),x1(:,2),'g.', 'LineWidth', 5);
scatter(x2(:,1),x2(:,2),'k.', 'LineWidth', 5);

for i=1:size(x,1)
    if t(i) > 0
        plot(x(i,1), x(i,2), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(i,1), x(i,2), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with ', num2str(size(k,2)), ' hidden layer, each layer with ', mat2str(k), ' hidden neuron']);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'red circle = \omega_2'},'FitBoxToText','on');

%% k = [2 2]

k = [2 2];
actfn = @(x) heaviside(x)*2-1;
[model, L] = mlpReg(x',y',k,actfn,lambda,maxiter);
t = mlpRegPred(model,actfn,x');
figure;
hold on
scatter(x1(:,1),x1(:,2),'g.', 'LineWidth', 5);
scatter(x2(:,1),x2(:,2),'k.', 'LineWidth', 5);

for i=1:size(x,1)
    if t(i) > 0
        plot(x(i,1), x(i,2), 'go','MarkerSize',10, 'LineWidth', 2)
    else
        plot(x(i,1), x(i,2), 'ko','MarkerSize',10, 'LineWidth', 2)
    end
end
hold off

title(['MLP with ', num2str(size(k,2)), ' hidden layer, each layer with ', mat2str(k), ' hidden neuron']);
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String', {'green circle = \omega_1', 'red circle = \omega_2'},'FitBoxToText','on');

%% Problem 4.1 Conclusion



%% Problem 4.2 Chapter 4 Page 240




