% Tai Duc Nguyen - CS 435 - Assignment 2 - 04/10/2019

%% Importing picture

clear all;
close all;
infile = 'other.jpg';
img = imread(infile);
img = rgb2gray(img);
imgdble = im2double(img);
[img_w, img_l] = size(img);
folder = 'C:\Users\b3nnyth3d3g\Dropbox\MATLAB_workspace\CS435_HW2';

%% Compute NxN smoothing kernel N=3 sigma=1

N = 3;
sigma = 1;

[dx, dy]=meshgrid(-1:1, -1:1);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Convolve image and kernel W

img_result1 = zeros(img_w, img_l);
k = floor(N/2);
img_pad = padarray(imgdble, double([k k]), 0);
for i = double(1+k):img_w
    for j = double(1+k):img_l
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        img_result1(x, y) = sum(sum(W.*T));
    end
end

%% Compute NxN smoothing kernel N=5 sigma=1

N = 5;
sigma = 1;

[dx, dy]=meshgrid(-2:2, -2:2);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Convolve image and kernel W

img_result2 = zeros(img_w, img_l);
k = floor(N/2);
img_pad = padarray(imgdble, double([k k]), 0);
for i = double(1+k):img_w
    for j = double(1+k):img_l
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        img_result2(x, y) = sum(sum(W.*T));
    end
end

%% Compute NxN smoothing kernel N=5 sigma=2

N = 5;
sigma = 2;

[dx, dy]=meshgrid(-2:2, -2:2);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Convolve image and kernel W

img_result3 = zeros(img_w, img_l);
k = floor(N/2);
img_pad = padarray(imgdble, double([k k]), 0);
for i = double(1+k):img_w
    for j = double(1+k):img_l
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        img_result3(x, y) = sum(sum(W.*T));
    end
end

%% Compute NxN smoothing kernel N=5 sigma=4

N = 5;
sigma = 4;

[dx, dy]=meshgrid(-2:2, -2:2);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Convolve image and kernel W

img_result4 = zeros(img_w, img_l);
k = floor(N/2);
img_pad = padarray(imgdble, double([k k]), 0);
for i = double(1+k):img_w
    for j = double(1+k):img_l
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        img_result4(x, y) = sum(sum(W.*T));
    end
end

%% Graph Part 1

figure(1)

subplot(2,3,1);
imshow(img)
title('Original')

subplot(2,3,2);
imshow(img_result1)
title('N=3, sigma=1')

subplot(2,3,3);
imshow(img_result2)
title('N=5, sigma=1')

subplot(2,3,4);
imshow(img_result3)
title('N=5, sigma=2')

subplot(2,3,5);
imshow(img_result4)
title('N=5, sigma=4')

%% Gradients respect to x

gx = zeros(img_w-2, img_l-2);
gy = zeros(img_w-2, img_l-2);

for i = 2:img_w-1
    for j = 2:img_l-1
        m = i - 1;
        n = j - 1;
        gx(m, n) = (img_result2(i + 1, j) - img_result2(i - 1, j))/2;
        gy(m, n) = (img_result2(i, j + 1) - img_result2(i, j - 1))/2;
    end
end

G = sqrt(gx.^2 + gy.^2);

%% Graph Part 2 Using gaussian filter N=5 sigma=2

figure(2)

subplot(2, 2, 1)
imshow(img_result2)
title('Original + filter N=5 sigma=1')

subplot(2, 2, 2)
imshow(gx)
title('Gradient with respect to x')

subplot(2, 2, 3)
imshow(gy)
title('Gradient with respect to y')

subplot(2, 2, 4)
imshow(G)
title('Combined Gradient')

%% Thresholding

T = 0.17;
bw1 = (G > T)*255;
T = 0.10;
bw2 = (G > T)*255;
T = 0.08;
bw3 = (G > T)*255;
T = 0.27;
bw4 = (G > T)*255;

figure(3)

subplot(2, 2, 1)
imshow(bw1)
title('threshold=0.17')

subplot(2, 2, 2)
imshow(bw2)
title('threshold=0.10')

subplot(2, 2, 3)
imshow(bw3)
title('threshold=0.08')

subplot(2, 2, 4)
imshow(bw4)
title('threshold=0.27')

%% Hyterisis

T1 = 0.23;
T2 = 0.13;

aboveT2 = G > T2;
[aboveT1row, aboveT1col] = find(img_result3 > T1);
bw2 = bwselect(aboveT2, aboveT1col, aboveT1row, 8);

figure(4)
imshow(bw2)
