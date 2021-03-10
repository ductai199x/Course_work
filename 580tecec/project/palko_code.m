%% ECEC-t580 Ultrasonic Transducer Filter
%% Jonathan Palko, Tai Nguyen

%% Alpha Filter
T = 1; %(time period between updates)

%Variances (Sigma^2) are for inches
var_w = 1/36;
var_n = 0.0058411; %10 inch variance based on class
% var_n = 0.005302; %10 inch data variance
sig_w = sqrt(var_w)
sig_n = sqrt(var_n)

%Only accounting for distance (inches)
data_10in = readtable('Kalman Filter Statistics (2 inch spacing).xlsx','Sheet','10"','Range','F1:F117');
data_10in = table2array(data_10in);

%Number of measured data points
datacount_10 = size(data_10in,1);

%Tracking idex for alpha filter
trackInd = (T^2)*(sig_w/sig_n)

alpha = ((-trackInd^2)+sqrt((trackInd^4)+16*trackInd^2))/8

%Position Error Variance P(k|k) = var_x
var_x = alpha*var_n
sig_x = sqrt(var_x)

%Average of measurements at 10 inches
x_10_ic = mean(data_10in)

%Initial position calculation
x_kplus1 = 0
x_kplus1(1,1) = x_10_ic + alpha*(data_10in(1,1)-x_10_ic)

for k = 2:datacount_10
   x_kplus1(k,1) = x_kplus1(k-1,1) + alpha*(data_10in(k,1)-x_kplus1(k-1,1));
end

xrange = 1:k;
yrange = x_kplus1;

figure(1)

%Plotting positions
stem(xrange,yrange);
axis([0 size(xrange,2) 9.8 10.2])
hold on;
grid on;
title('10 inch Alpha Filter');
ylabel('Position (inch)');
xlabel('Sample Number (k)');

%Plotting performance levels
avg_pos = mean(x_kplus1)*ones(size(yrange));
up_bound = avg_pos + 3*sig_x;
low_bound = avg_pos - 3*sig_x;
plot(xrange,avg_pos,'--');
plot(xrange,up_bound,'g--');
plot(xrange,low_bound,'m--');
legend('Alpha-Filter Data','Average Value','+3 sigma_x','-3 sigma_x');

%% Alpha-Beta Implementation
% vel = 17.6; %~ 1mph
vel = 1.5; %~ 0.0852273mph
T2 = 2/vel; %2inch increments (amount of time to move 2 inches and measure)

%Solving for alpha and beta
track = (T2^2*sig_w)/sig_n;
syms a b
rightalp = ((2*(2-a)-4*sqrt(1-a))^2)/(1-a);
testalp = vpa(solve(rightalp==track^2,a),6);
testalp(imag(testalp)~=0)= [];
testalp(testalp<=0) = [];

alp2 = testalp;
% alp2 = 0.21115;

bet2 = 2*(2-alp2)-4*sqrt(1-alp2);
% bet2 = 0.025011;

K = [alp2;bet2/T2];
trans = [1 T2;0 1];
h = [1 0];
x_data = [4.8976    7.0081    9.0105   11.2293   12.8257   14.6657   16.7763   18.5892   20.3480   22.5668   24.9750   26.5444   28.4114   30.6843   32.3890   34.4996];
var_v = var_n*(((2*alp2-bet2)*bet2)/(2*(1-alp2)*T2^2));
sig_v = sqrt(var_v);
IC = [5;vel];
datacount = size(x_data,2);

%Initial position calculation
y = IC + K*[x_data(1)-h*IC];
y = trans*y;

for k = 2:datacount
   y(:,k) = y(:,k-1) + K*[x_data(k)-h*y(:,k-1)];
   y(:,k) = trans*y(:,k);
end

xrange2 = 1:k;
yrange2 = y;

figure(2)

%Plotting positions
stem(xrange2,yrange2(1,:));
% axis([0 size(xrange2,2) 9.8 10.2])
hold on;
grid on;
title('Alpha-Beta Filter Position');
ylabel('Position (inch)');
xlabel('Sample Number (k)');

figure (3)

%Plotting velocities
stem(xrange2,yrange2(2,:));
% axis([0 size(xrange2,2) 15.5 17.5])
hold on;
grid on;
title('Alpha-Beta Filter Velocity');
ylabel('Velocity (inch/s)');
xlabel('Sample Number (k)');

%Plotting performance levels
avg_pos2 = mean(y(2,:))*ones(size(yrange2(2,:)));
up_bound2 = avg_pos2 + 3*sig_v;
low_bound2 = avg_pos2 - 3*sig_v;
plot(xrange2,avg_pos2,'--');
plot(xrange2,up_bound2,'g--');
plot(xrange2,low_bound2,'m--');
legend('Alpha-Beta Filter Data','Average Value','+3 sigma_v','-3 sigma_v');
