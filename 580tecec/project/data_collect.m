%% Initialize
clear; close; clc;

% Get information about connected devices
info = instrhwinfo('serial')
serial_port = info.SerialPorts(2);


%% 
% Open a serial connection to arduino
mega = serial(serial_port{1},'BaudRate',19200);
try
    fopen(mega);
catch
    fprintf('cant open\n');
    fclose(mega);
end

%%

pause(1);
L=20;
x = 0:L;
y = zeros(1, L);
for K = 0 : L
    flushinput(mega);
    fprintf(mega, "2");
    pause(0.1);
    y(K + 1) =  str2double(fscanf(mega));
    flushinput(mega);
    disp([K, y(K + 1)]);
end

fclose(mega);

delete(mega);
clear mega;

figure
plot(x, y, 'bo', 'MarkerFaceColor', 'blue');
grid on
xlabel('t (%)');
ylabel('dist (cm)');