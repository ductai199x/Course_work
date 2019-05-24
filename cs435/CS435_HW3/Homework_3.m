% Tai Duc Nguyen - CS 435 - Assignment 3 - 05/04/2019

%% Import images

clear all;
close all;
file1 = 'boat.jpg';
img1 = imread(file1);
imgdble1 = im2double(img1);

file2 = 'fortress.jpg';
img2 = imread(file2);
imgdble2 = im2double(img2);

%% Part 1: Crop and Rescale

% Algorithm is written after Richard Alan Peters' II digital image 
% processing slides on interpolation
% https://ia902707.us.archive.org/23/items/Lectures_on_Image_Processing/EECE_4353_15_Resampling.pdf

figure(1)

h=[];
im = imgdble1;
h(1) = subplot(2, 4, 1);
image(im, 'Parent', h(1))
title('Original boat.jpg')

scale = [2 2];
outimg = bilinearInterpolation(im, scale);
h(2) = subplot(2, 4, 2);
image(outimg, 'Parent', h(2))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])

scale = [0.5 0.5];
outimg = bilinearInterpolation(im, scale);
h(3) = subplot(2, 4, 3);
image(outimg, 'Parent', h(3))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])

scale = [0.7 1.9];
outimg = bilinearInterpolation(im, scale);
h(4) = subplot(2, 4, 4);
image(outimg, 'Parent', h(4))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])


im = imgdble2;
h(5) = subplot(2, 4, 5);
image(im, 'Parent', h(5))
title('Original fortress.jpg')

scale = [1.5 1.5];
outimg = bilinearInterpolation(im, scale);
h(6) = subplot(2, 4, 6);
image(outimg, 'Parent', h(6))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])

scale = [0.7 0.7];
outimg = bilinearInterpolation(im, scale);
h(7) = subplot(2, 4, 7);
image(outimg, 'Parent', h(7))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])

scale = [0.3 0.7];
outimg = bilinearInterpolation(im, scale);
h(8) = subplot(2, 4, 8);
image(outimg, 'Parent', h(8))
title(['[height scale, width scale] = [', num2str(scale(1)), ',', num2str(scale(2)), ']'])

axis image


%% Part 2: Energy Function

% NxN smoothing kernel N=5 sigma=1
N = 5;
sigma = 1;

[dx, dy]=meshgrid(-2:2, -2:2);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Image 1: boat.jpg
im = rgb2gray(imgdble1);

im_smooth = zeros(size(im,1), size(im,2));
k = floor(N/2);
img_pad = padarray(im, double([k k]), 0);
for i = double(1+k):size(im,1)
    for j = double(1+k):size(im,2)
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        im_smooth(x, y) = sum(sum(W.*T));
    end
end

% Gradients

gx = zeros(size(im,1)-2, size(im,2)-2);
gy = zeros(size(im,1)-2, size(im,2)-2);

for i = 2:size(im,1)-1
    for j = 2:size(im,2)-1
        m = i - 1;
        n = j - 1;
        gx(m, n) = (im_smooth(i + 1, j) - im_smooth(i - 1, j))/2;
        gy(m, n) = (im_smooth(i, j + 1) - im_smooth(i, j - 1))/2;
    end
end

E1 = sqrt(gx.^2 + gy.^2);
E1 = padarray(E1, double([1 1]), 'replicate');
E1(1,:) = E1(2,:);
E1(size(E1,1),:) = E1(size(E1,1)-1,:);

figure(2)
subplot(2, 2, 1)
imshow(im)
title('Original image boat.jpg')

subplot(2, 2, 2)
imshow(E1)
title('Energy Function boat.jpg')


%% Image 2: fortress.jpg
im = rgb2gray(imgdble2);

im_smooth = zeros(size(im,1), size(im,2));
k = floor(N/2);
img_pad = padarray(im, double([k k]), 0);
for i = double(1+k):size(im,1)
    for j = double(1+k):size(im,2)
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        im_smooth(x, y) = sum(sum(W.*T));
    end
end

% Gradients

gx = zeros(size(im,1)-2, size(im,2)-2);
gy = zeros(size(im,1)-2, size(im,2)-2);

for i = 2:size(im,1)-1
    for j = 2:size(im,2)-1
        m = i - 1;
        n = j - 1;
        gx(m, n) = (im_smooth(i + 1, j) - im_smooth(i - 1, j))/2;
        gy(m, n) = (im_smooth(i, j + 1) - im_smooth(i, j - 1))/2;
    end
end

E2 = sqrt(gx.^2 + gy.^2);
E2 = padarray(E2, double([1 1]), 'replicate');
E2(1,:) = E2(2,:);
E2(size(E2,1),:) = E2(size(E2,1)-1,:);

subplot(2, 2, 3)
imshow(im)
title('Original image boat.jpg')

subplot(2, 2, 4)
imshow(E2)
title('Energy Function boat.jpg')

%% Optimal Seam

M = zeros(size(E1,1), size(E1,2));

for i=1:size(E1,1)
    M = getOptimalSeam(E1, M, i);
end

P = backtrack(M);
im = imgdble1;
im(:,:,1) = im(:,:,1) + P;

figure(3)
subplot(1,2,1)
imshow(im)
title('Optimal Seam boat.jpg')

M = zeros(size(E2,1), size(E2,2));

for i=1:size(E2,1)
    M = getOptimalSeam(E1, M, i);
end

P = backtrack(M);
im = imgdble2;
im(:,:,1) = im(:,:,1) + P;

figure(3)
subplot(1,2,2)
imshow(im)
title('Optimal Seam fortress.jpg')

%% Seam Carving

num_seam = 50;

%% For boat.jpg

im = imgdble1;
E = E1;
frames = cell(num_seam,1);

for i=1:num_seam
    
    % Get seam
    M = zeros(size(E,1), size(E,2));
    for j=1:size(E,1)
        M = getOptimalSeam(E, M, j);
    end

    P = backtrack(M);
    im(:,:,1) = im(:,:,1) + P;
    tmp = im;
    tmp(tmp>1) = 1;
    
    % Add img to frames
    frames{i} = tmp;
    
    % Remove seam
    P = (-1)*P;
    P(P==0) = 1;

    for c=1:3
        chan = im(:,:,c);
        chan = (chan + 1) .* P;
        for k=1:size(chan,1)
            chan(k,:) = [chan(k, 1:find(chan(k,:)<0)-1)-1, chan(k,find(chan(k,:)<0)+1:end)-1, 0];
        end
        im(:,:,c) = chan;
    end

    E = (E+1) .* P;
    for k=1:size(E,1)
        E(k,:) = [E(k, 1:find(E(k,:)<0)-1)-1, E(k,find(E(k,:)<0)+1:end)-1, 100];
    end

end

 % create the video writer with 1 fps
 writerObj = VideoWriter('boat.avi');
 writerObj.FrameRate = 1;
 % set the seconds per image
 secsPerImage = ones(1, num_seam);
 % open the video writer
 open(writerObj);
 % write the frames to the video
 for u=1:num_seam
     % convert the image to a frame
     p = size(frames{1},2) - size(frames{u},2);
     frame = im2frame(padarray(frames{u}, [0 p 0], 0, 'post'));
     for v=1:secsPerImage(u) 
         writeVideo(writerObj, frame);
     end
 end
 % close the writer object
 close(writerObj);

%% For fortress.jpg

im = imgdble2;
E = E2;
frames = cell(num_seam,1);

for i=1:num_seam
    
    % Get seam
    M = zeros(size(E,1), size(E,2));
    for j=1:size(E,1)
        M = getOptimalSeam(E, M, j);
    end

    P = backtrack(M);
    im(:,:,1) = im(:,:,1) + P;
    tmp = im;
    tmp(tmp>1) = 1;
    
    % Add img to frames
    frames{i} = tmp;
    
    % Remove seam
    P = (-1)*P;
    P(P==0) = 1;

    for c=1:3
        chan = im(:,:,c);
        chan = (chan + 1) .* P;
        for k=1:size(chan,1)
            chan(k,:) = [chan(k, 1:find(chan(k,:)<0)-1)-1, chan(k,find(chan(k,:)<0)+1:end)-1, 0];
        end
        im(:,:,c) = chan;
    end

    E = (E+1) .* P;
    for k=1:size(E,1)
        E(k,:) = [E(k, 1:find(E(k,:)<0)-1)-1, E(k,find(E(k,:)<0)+1:end)-1, 100];
    end

end

 % create the video writer with 1 fps
 writerObj = VideoWriter('fortress.avi');
 writerObj.FrameRate = 1;
 % set the seconds per image
 secsPerImage = ones(1, num_seam);
 % open the video writer
 open(writerObj);
 % write the frames to the video
 for u=1:num_seam
     % convert the image to a frame
     p = size(frames{1},2) - size(frames{u},2);
     frame = im2frame(padarray(frames{u}, [0 p 0], 0, 'post'));
     for v=1:secsPerImage(u) 
         writeVideo(writerObj, frame);
     end
 end
 % close the writer object
 close(writerObj);












