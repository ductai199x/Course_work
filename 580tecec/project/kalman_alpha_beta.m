%% Tai Duc Nguyen - ECEC T580 - HW3

clear all; close all;

%% Kalman Filter Target Tracker

rng(0);

%% Create ground truth and measurement

dataset = xlsread('data.xlsx', 2, 'A1:Q104');

x_gt = [5, 7, 9, 11.25, 13, 14.75, 17, 19, 20.75, 23, 25.25, 27, 29, 31.25, 32.75, 34.75];

% 1 mile = 63360 inches
% 63360 inches per hour
avg_velocity = 10000/3600; % inches per second
v_gt = ones(1,length(x_gt))*avg_velocity;
total_travel_time = (x_gt(end) - x_gt(1))/10000*3600; % in seconds
dt = total_travel_time/(length(x_gt)-1);
t = linspace(0, total_travel_time, length(x_gt));

% dataset_row = randi([1 size(dataset, 1)]);
dataset_row = 64;
x_measure = dataset(dataset_row,:);

% x_measure = max(dataset);

x_avg = mean(dataset(randi([1 size(dataset, 1)], [1, 10]), :));


%% Alpha-beta filter

order = 2;
var_w = 1/36;
var_n = 0.0058411; %10 inch variance based on class
% var_n = 0.03;

sig_w = sqrt(var_w);
sig_n = sqrt(var_n);

A = [1 dt; 0 1];
B = [(dt^2/2); dt];
% B = [0; dt];
H = [1; 0];

% get the optimal alpha and beta values
lambda = (dt^2*sig_w)/sig_n;
r = (4 + lambda - sqrt(8*lambda + lambda^2))/4;

alpha = 1 - r^2;
beta = 2*(2-alpha) - 4*sqrt(1-alpha);

xp_init = [x_gt(1); avg_velocity];
PC_kalata = [1 1; 1 2+lambda^2/4].*var_n;
% PC_init = 1e3*eye(order);
PC_init = PC_kalata;


R = var_w;
QA = B*B'* (var_w);

P_corrected = PC_init;
P_pred = A*P_corrected*A' + QA;
x_pred = xp_init;

x_hist = zeros(2, length(t));

K = [alpha; beta/dt];
for i = 1:length(t)
    x_corrected = x_pred + K*(x_measure(i) - H'*x_pred);
    x_hist(:,i) = x_corrected;
    P_corrected = (eye(order) - K*H')*P_pred;
    x_pred = A*x_corrected;
    P_pred = A*P_corrected*A' + QA;
end


% for i = 1:length(t)
%     K = P_pred*H*inv(H'*P_pred*H + R);
%     x_corrected = x_pred + K*(x_measure(i) - H'*x_pred);
%     x_hist(:,i) = x_corrected;
%     P_corrected = (eye(order) - K*H')*P_pred;
%     x_pred = A*x_corrected;
%     P_pred = A*P_corrected*A' + QA;   
% end


var_v = ((2*alpha - beta)*beta)/(2*(1 - alpha)*dt^2)*var_n;
sigma_v = sqrt(var_v);

var_x = alpha*var_n;
sigma_x = sqrt(var_x);

figure(1)
hold on
plot(t, x_gt, "og");
plot(t, x_measure, ".r");
stem(t, x_hist(1,:), "*b");
xlabel("Time (s)");
ylabel("Position (inch)");
legend("ground truth", "measured", "corrected");
title("Position ground truth vs alpha-beta filter output");
hold off

figure(2)
hold on
plot(t, x_gt-x_hist(1,:), "ok");
plot(t, x_gt-x_avg, "xb");
plot(t, zeros(length(x_gt)));
plot(t, 3*sigma_x*ones(length(x_gt)), "--r");
plot(t, -3*sigma_x*ones(length(x_gt)), "--g");
xlabel("Time (s)");
ylabel("Position (inch)");
legend("KF error", "10-pt-average error", "mean", "+3 sigma", "-3 sigma");
title("Error between position ground truth vs alpha-beta filter output");
hold off

figure(3)
hold on
plot(t, v_gt, "-g");
stem(t, x_hist(2,:));
plot(t, v_gt + 3*sigma_v*ones(length(v_gt)), "--r");
plot(t, v_gt - 3*sigma_v*ones(length(v_gt)), "--b");
xlabel("Time (s)");
ylabel("Velocity (inch/second)");
legend("ground truth", "filtered", "+3 sigma", "-3 sigma");
title("Velocity ground truth vs alpha-beta filter output");
hold off

figure(4)
hold on
plot(x_hist(1,:), x_hist(2,:), 'or')
plot(x_gt, v_gt, 'xk')
rho = 0.95;  % 95% confidence
color = 'r';
for i = 1:length(t)
    plotErrorEllipse(x_hist(:,i), P_corrected, rho, color);
%     plotErrorEllipse([x_gt(i), v_gt(i)], P_corrected, rho, color);
end
rho = 0.90;  % 95% confidence
color = 'b';
for i = 1:length(t)
    plotErrorEllipse(x_hist(:,i), P_corrected, rho, color);
%     plotErrorEllipse([x_gt(i), v_gt(i)], P_corrected, rho, color);
end

ylim([avg_velocity-9*sigma_v, avg_velocity+9*sigma_v])
xlim([min(x_gt)-0.75 max(x_gt)+0.75])
xlabel("Position (inch)");
ylabel("Velocity (inch/second)");
legend("KF output", "ground truth");
title("Phase Plant Plot for KF Output");
hold off
