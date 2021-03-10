%% Tai Duc Nguyen - ECE 303 - Lab 4

clear all; close all;

%%

base_white = load('base_white.mat').y/1024*5;
base_red = load("base_red.mat").y/1024*5;
base_green = load("base_green.mat").y/1024*5;
white_5k = load("5k_white.mat").y/1024*5;
white_5avg = load("white_5avg.mat").y/1024*5;


duty_cycle = 0:255;
duty_cycle = duty_cycle/255*100;

r = 10000;

%% baseline white 10k

figure(1)

r_volt = base_white;
photo_volt = 5-r_volt;
current = r_volt/r;
photo_r = photo_volt./current;

subplot(1,2,1)
hold on
plot(duty_cycle, r_volt, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
plot(duty_cycle, photo_volt, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
legend("resistor voltage", "photocell voltage");
xlabel("% duty cycle");
ylabel("Voltage (V)");
title("Baseline WHITE 10k");
ylim([0 5])
hold off

subplot(1,2,2)
yyaxis left
plot(duty_cycle, photo_r, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
ylabel("Resistance (Ohms)");

yyaxis right
plot(duty_cycle, current*1000, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
ylabel("Current (mA)");
legend("photocell resistance", "current");
xlabel("% duty cycle");
title("Baseline WHITE 10k");

%% baseline red 10k

figure(2)

r_volt = base_red;
photo_volt = 5-r_volt;
current = r_volt/r;
photo_r = photo_volt./current;

subplot(1,2,1)
hold on
plot(duty_cycle, r_volt, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
plot(duty_cycle, photo_volt, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
legend("resistor voltage", "photocell voltage");
xlabel("% duty cycle");
ylabel("Voltage (V)");
title("Baseline RED 10k");
ylim([0 5])
hold off

subplot(1,2,2)
yyaxis left
plot(duty_cycle, photo_r, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
ylabel("Resistance (Ohms)");

yyaxis right
plot(duty_cycle, current*1000, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
ylabel("Current (mA)");
legend("photocell resistance", "current");
xlabel("% duty cycle");
title("Baseline RED 10k");

%% baseline green 10k

figure(3)

r_volt = base_green;
photo_volt = 5-r_volt;
current = r_volt/r;
photo_r = photo_volt./current;

subplot(1,2,1)
hold on
plot(duty_cycle, r_volt, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
plot(duty_cycle, photo_volt, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
legend("resistor voltage", "photocell voltage");
xlabel("% duty cycle");
ylabel("Voltage (V)");
title("Baseline GREEN 10k");
ylim([0 5])
hold off

subplot(1,2,2)
yyaxis left
plot(duty_cycle, photo_r, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
ylabel("Resistance (Ohms)");

yyaxis right
plot(duty_cycle, current*1000, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
ylabel("Current (mA)");
legend("photocell resistance", "current");
xlabel("% duty cycle");
title("Baseline GREEN 10k");

%% white 5k

figure(4)

r_volt = white_5k;
photo_volt = 5-r_volt;
current = r_volt/r;
photo_r = photo_volt./current;

subplot(1,2,1)
hold on
plot(duty_cycle, r_volt, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
plot(duty_cycle, photo_volt, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
legend("resistor voltage", "photocell voltage");
xlabel("% duty cycle");
ylabel("Voltage (V)");
title("WHITE 5k");
ylim([0 5])
hold off

subplot(1,2,2)
yyaxis left
plot(duty_cycle, photo_r, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
ylabel("Resistance (Ohms)");

yyaxis right
plot(duty_cycle, current*1000, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
ylabel("Current (mA)");
legend("photocell resistance", "current");
xlabel("% duty cycle");
title("WHITE 5k");

%% white 5avg

figure(5)

r_volt = white_5avg;
photo_volt = 5-r_volt;
current = r_volt/r;
photo_r = photo_volt./current;

subplot(1,2,1)
hold on
plot(duty_cycle, r_volt, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
plot(duty_cycle, photo_volt, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
legend("resistor voltage", "photocell voltage");
xlabel("% duty cycle");
ylabel("Voltage (V)");
title("WHITE 10k 5avg");
ylim([0 5])
hold off

subplot(1,2,2)
yyaxis left
plot(duty_cycle, photo_r, 'bo', 'MarkerFaceColor', 'blue', 'MarkerSize', 3);
ylabel("Resistance (Ohms)");

yyaxis right
plot(duty_cycle, current*1000, 'ro', 'MarkerFaceColor', 'red', 'MarkerSize', 3);
ylabel("Current (mA)");
legend("photocell resistance", "current");
xlabel("% duty cycle");
title("WHITE 10k 5avg");

