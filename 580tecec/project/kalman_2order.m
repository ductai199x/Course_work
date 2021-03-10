%% Tai Duc Nguyen - ECEC T580 - HW3

clear all; close all;

%% Kalman Filter Target Tracker

rng(0);

%% Create ground truth and measurement

max_time = 2000;
dt = 1;

a_gt = zeros(1,max_time);
a_gt(600:800) = 32.2;
a_gt(1000:1200) = -32.2;

v_gt = cumsum(a_gt)*dt;
x_gt = cumsum(v_gt)*dt;

sigma_x = 200; mu_x = 0;
noise_measure = normrnd(mu_x,sigma_x, [1,max_time]);
x_measure = x_gt + noise_measure;

%% Create Kalman Filter tracking position and velocity

order = 2;
A = [1 dt; 0 1];
B = [(dt^2/2); dt];
H = [1; 0];

PC_init = 1e6*eye(order);
xp_init = [0;0];
R = sigma_x^2;
QA = B*B'* (32.2^2);

P_corrected = PC_init;
P_pred = A*P_corrected*A' + QA;
x_pred = xp_init;

for i = 1:max_time
    K = P_pred*H*inv(H'*P_pred*H + R);
    x_corrected = x_pred + K*(x_measure(i) - H'*x_pred);
    P_corrected = (eye(order) - K*H')*P_pred;
    x_pred = A*x_corrected;
    P_pred = A*P_corrected*A' + QA;   
end

disp(P_corrected);
disp(P_pred);
disp(K);

