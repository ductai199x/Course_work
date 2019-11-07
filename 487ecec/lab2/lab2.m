
%%
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
X = [X ones(N*2,1)];
Y = [ones(N,1); ones(N,1)*-1];

figure
scatter(X1(:,1), X1(:,2), 'r.');
hold on
scatter(X2(:,1), X2(:,2), 'k.');

slope = rand() * 1000;

w0 = 0;
w1 = rand()*100000;
w2 = rand()*100000;

W = [w1 w2 w0];

decision_x = linspace(min(X(:,1)), max(X(:,1)), 10000);
decision_y = -(w1/w2)*decision_x - (w0/w2);

plot(decision_x, decision_y, "k");

for i=1:size(X,1)
   if sum(W.*X(i,:).*Y(i)) < 0
       plot(X(i,1),X(i,2),'og');
   end
    
end


accuracy = sum(W*X'.*Y' > 0)/size(X,1)

%%

M1 = [0 0];
M2 = [0 100];

X1 = mvnrnd(M1, S, N);
X2 = mvnrnd(M2, S, N);

X = [X1;X2];
X = [X ones(N*2,1)];
Y = [ones(N,1); ones(N,1)*-1];

figure
scatter(X1(:,1), X1(:,2), 'r.');
hold on
scatter(X2(:,1), X2(:,2), 'k.');

slope = rand() * 1000;

w0 = 0;
w1 = rand()*100000;
w2 = rand()*100000;

W = [w1 w2 w0];

decision_x = linspace(min(X(:,1)), max(X(:,1)), 10000);
decision_y = -(w1/w2)*decision_x - (w0/w2) + 50;

plot(decision_x, decision_y, "k");

for i=1:size(X,1)
   if sum(W.*X(i,:).*Y(i)) + 50 < 0
       plot(X(i,1),X(i,2),'og');
   end
    
end


accuracy = sum(W*X'.*Y' > 0)/size(X,1)