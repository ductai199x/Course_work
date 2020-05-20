%% Initialize
clear; close; clc;

% Get information about connected devices
info = instrhwinfo('serial')
serial_port = info.SerialPorts(2);


%% 
% Open a serial connection to arduino
mega = serial(serial_port{1},'BaudRate',9600);
try
    fopen(mega);
catch
    fprintf('cant open\n');
    fclose(mega);
end

%%

pause(1);
x = 0:255;
y = zeros(1, 256);
for K = 0 : 255
    flushinput(mega);
    fprintf(mega, "2");
    pause(0.1);
    y(K + 1) =  str2double(fscanf(mega))/5;
    flushinput(mega);
    disp([K, y(K + 1)]);
end

fclose(mega);

delete(mega);
clear mega;

figure
plot(x / 255 * 100, y * 5 / 1023, 'bo', 'MarkerFaceColor', 'blue');
grid on
xlabel('Duty Cycle (%)');
ylabel('Output Voltage (V)');