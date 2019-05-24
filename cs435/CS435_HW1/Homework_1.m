% Tai Duc Nguyen - CS 435 - Assignment 1 - 04/03/2019

%% Importing picture

clear all;
close all;
infile = 'golden_retriever.jpg';
img = imread(infile);
imgdble = im2double(img);
folder = 'C:\Users\b3nnyth3d3g\Dropbox\MATLAB_workspace\CS435_HW1';

%% Part 1: RGB -> Gray

gray_img = 0.2989*imgdble(:,:,1) + 0.5870*imgdble(:,:,2) + 0.1140*imgdble(:,:,3);
path = sprintf('%s\\gray_%s', folder, infile);
imwrite(gray_img, path);

%% Part 2: RGB -> Binary

threshold = 0.25;
bin25_img = zeros(size(gray_img));
bin25_img(gray_img > threshold) = 1;
path = sprintf('%s\\bin_25_%s', folder, infile);
imwrite(bin25_img, path);

threshold = 0.5;
bin50_img = zeros(size(gray_img));
bin50_img(gray_img > threshold) = 1;
path = sprintf('%s\\bin_50_%s', folder, infile);
imwrite(bin50_img, path);

threshold = 0.75;
bin75_img = zeros(size(gray_img));
bin75_img(gray_img > threshold) = 1;
path = sprintf('%s\\bin_75_%s', folder, infile);
imwrite(bin75_img, path);

%% Part 3: Histogram

close();
figure(1);

% gray scale
gray_img_int = 0.2989*img(:,:,1) + 0.5870*img(:,:,2) + 0.1140*img(:,:,3);
bins_gray = zeros(1, 256);
flatX = reshape(gray_img_int,1,numel(gray_img_int));
for val = 0:255
    bins_gray(val+1) = sum(flatX==val);
end
bins_gray = bins_gray/sum(bins_gray);

subplot(2,2,1);
bar(bins_gray)
title('Distribution of pixels in grayscale');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

% red channel
bins_red = zeros(1, 256);
flatX = reshape(img(:,:,1),1,numel(img(:,:,1)));
for val = 0:255
    bins_red(val+1) = sum(flatX==val);
end
bins_red = bins_red/sum(bins_red);

subplot(2,2,2);
bar(bins_red)
title('Distribution of pixels in red channel');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

% green channel
bins_green = zeros(1, 256);
flatX = reshape(img(:,:,2),1,numel(img(:,:,2)));
for val = 0:255
    bins_green(val+1) = sum(flatX==val);
end
bins_green = bins_green/sum(bins_green);

subplot(2,2,3);
bar(bins_green)
title('Distribution of pixels in green channel');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

% blue channel
bins_blue = zeros(1, 256);
flatX = reshape(img(:,:,3),1,numel(img(:,:,3)));
for val = 0:255
    bins_blue(val+1) = sum(flatX==val);
end
bins_blue = bins_blue/sum(bins_blue);

subplot(2,2,4);
bar(bins_blue)
title('Distribution of pixels in blue channel');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

%% Part 4: Contrast Stretching

figure(2);

% Diming the image
dimmed_img = gray_img_int./2;
flat_gray = reshape(dimmed_img,1,numel(dimmed_img));

% Histogram of the dimmed image
bins_cs = zeros(1, 256);
for val = 0:255
    bins_cs(val+1) = sum(flat_gray==val);
end
bins_cs = bins_cs/sum(bins_cs);
subplot(2,2,1)
bar(bins_cs)
title('Distribution of pixels in dimmed img (1/2 brightness)');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

% Show the image
subplot(2,2,3)
imshow(dimmed_img)
title('Dimmed image');

% Contrast stretching
r1 = 0;     s1 = 0;
r2 = 68;    s2 = 100;
r3 = 100;    s3 = 160;
r4 = 123;    s4 = 200;
r5 = max(flat_gray);    s5 = 255;

flat_cs = uint8(zeros(size(flat_gray)));

% Stretch region 0->40 into 0->70
temp = r1 <= flat_gray & r2 > flat_gray;
temp = uint8(temp).*flat_gray;
temp = (temp-r1)*((s2-s1)/(r2-r1));
temp(temp>0) = temp(temp>0) + s1;
flat_cs = flat_cs + temp;

% Stretch region 40->64 into 70->110
temp = r2 <= flat_gray & r3 > flat_gray;
temp = uint8(temp).*flat_gray;
temp = (temp-r2)*((s3-s2)/(r3-r2));
temp(temp>0) = temp(temp>0) + s2;
flat_cs = flat_cs + temp;

% Stretch region 64->75 into 110->130
temp = r3 <= flat_gray & r4 > flat_gray;
temp = uint8(temp).*flat_gray;
temp = (temp-r3)*((s4-s3)/(r4-r3));
temp(temp>0) = temp(temp>0) + s3;
flat_cs = flat_cs + temp;

% Stretch region 64->75 into 110->130
temp = r4 <= flat_gray & r5 >= flat_gray;
temp = uint8(temp).*flat_gray;
temp = (temp-r4)*((s5-s4)/(r5-r4));
temp(temp>0) = temp(temp>0) + s4;
flat_cs = flat_cs + temp;

% Histogram of the stretched image
bins_cs = zeros(1, 256);
for val = 0:255
    bins_cs(val+1) = sum(flat_cs==val);
end
bins_cs = bins_cs/sum(bins_cs);
subplot(2,2,2)
bar(bins_cs)
title('Distribution of pixels in contrast stretched img');
xlabel('Pixel value');
ylabel('Occurrence (normalized)');

cs_img = reshape(flat_cs, 732, 1100);
% Show the image
subplot(2,2,4)
imshow(cs_img)
title('Contrast stretched image');

% Save processed image to file
path = sprintf('%s\\contrast_%s', folder, infile);
imwrite(cs_img, path);




