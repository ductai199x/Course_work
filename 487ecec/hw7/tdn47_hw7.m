%% Tai Duc Nguyen - ECEC487 - 11/10/2019

clear all; close all;
rng(0);

%% Problem 6.1 Textbook page 401-402

l = 2;
N = 1000;
a = 10; e = 1;

w = [1 1];
w0 = 0;
X = generate_hyper(w',w0, a, e, N);

[coeffs, score, latent] = pca(X');

pc1 = [0 0; coeffs(:,1)'*1000];

hold on
plot(pc1(:,1), pc1(:,2), 'r');


%% Using generating points around (l-1) dimenstional hyperplane procedures in textbook page 399
function X = generate_hyper(w,w0,a,e,N)
    l=length(w);
    t=(rand(l-1,N)-.5)*2*a;
    t_last=-(w(1:l-1)/w(l))'*t + 2*e*(rand(1,N)-.5)-(w0/w(l));
    X=[t; t_last];
    %Plots for the 2d and 3d case
    if(l==2)
        figure(1), plot(X(1,:),X(2,:),'.b')
    elseif(l==3)
        figure(1), plot3(X(1,:),X(2,:),X(3,:),'.b')
    end
    figure(1)
    xlim([-a a]); ylim([-a a]);
end

