clear all
close all

a = 0:100;

r1 = 0;     s1 = 0;
r2 = 20;    s2 = 10;
r3 = 25;    s3 = 95;
r4 = 100;   s4 = 100;

b = zeros(size(a));

temp = r1 <= a & r2 > a;
temp = double(temp).*a;
temp = (temp-r1)*((s2-s1)/(r2-r1));
temp(temp>=0) = temp(temp>=0) + s1;
temp(temp<0) = 0;
b = b + temp;

temp = r2 <= a & r3 > a;
temp = double(temp).*a;
temp = (temp-r2)*((s3-s2)/(r3-r2));
temp(temp>=0) = temp(temp>=0) + s2;
temp(temp<0) = 0;
b = b + temp;

temp = r3 <= a & r4 >= a;
temp = double(temp).*a;
temp = (temp-r3)*((s4-s3)/(r4-r3));
temp(temp>=0) = temp(temp>=0) + s3;
temp(temp<0) = 0;
b = b + temp;

figure(1)
plot(a, b)
xlabel('Input value')
ylabel('Output value')