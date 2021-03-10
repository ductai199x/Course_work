%% Initialize
clear; close; clc;

% Get information about connected devices
info = instrhwinfo('serial');
serial_port = info.SerialPorts(2);

% Open a serial connection to arduino
mega = serial(serial_port{1},'BaudRate',9600);
try
    fopen(mega);
catch
    fprintf('cant open\n');
    fclose(mega);
end

%% Collect data

n_points = 100;
v_in = linspace(0, 5, n_points)';
v_adc = zeros(size(v_in));

for i = 1:n_points
    
    % Set Vin (via PWM --> LPF)
    scpi_str = sprintf(':SOURCE:VOLTAGE %3.2fV', v_in(i));
    fprintf(mega, scpi_str); 
    
    % Give the LPF's capacitor some time to charge
    pause(0.5);
    
    % Query the ADC voltage, convert result to double
    fprintf(mega, ':MEASURE:VOLTAGE?');
    v_adc(i) = str2double(fscanf(mega));
    
    fprintf('v_in = %3.2f, v_adc = %6.5f\n', v_in(i), v_adc(i));
end

% Reset voltage to zero
% fprintf(mega, ':SOURCE:VOLTAGE 0V');

%% Plot
figure(1)
scatter(v_in, v_adc);
xlabel('Input Voltage V_{in}');
ylabel('ADC Voltage V_{ADC}');
grid on;

%% Convert ADC voltage to current

% Your code goes here. Convert the voltage read from the ADC to current
% through your load resistor as a function of v_adc and Rs.
Rs = 9.4;
i_L = v_adc_/Rs;

%% Plot the regression line over the data
% When you have an array of currents i_L for each v_in, perform a linear
% regression. If you treat v_in as the independent variable, and i_L as the
% dependent, the slope of the regression line is your estimate of the 
% 'unknown' load admittance or 1/RL. Swapping the independent and dependent
% variables gives a slope equal to the estimated resistance RL.

temp = (i_L(i_L>0))';
X = [ ones(length(temp),1) temp(:) ];
a = (X.'*X)\(X.'*(v_in(i_L > 0)))
b = a(1); m = a(2);
x = linspace(min(temp), max(temp), length(temp));
f = b + m*x;
figure(2)
scatter(i_L, v_in);
hold on
plot(x, f, '-r');
hold off;
xlabel('Input Voltage V_{in}');
ylabel('Measured Load Current I_{L}');
title(sprintf('Estimated R_L = %.1f', m));
grid on;

%% Don't forget to close the port

fclose(mega);