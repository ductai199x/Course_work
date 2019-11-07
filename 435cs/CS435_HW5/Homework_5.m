% Tai Duc Nguyen - CS 435 - Assignment 5 - 05/29/2019

%% Importing picture

clear all;
close all;
infile = 'test2.jpg';
img = imread(infile);
img_gray = rgb2gray(img);
imgdble = im2double(img_gray);
[img_w, img_l] = size(img_gray);

%% Compute NxN smoothing kernel N=5 sigma=2

N = 5;
sigma = 2;

[dx, dy]=meshgrid(-2:2, -2:2);
W = exp( -dx.^2 / (2*sigma^2) - dy.^2 / (2*sigma^2) );
W = W./sum(W(:));

%% Convolve image and kernel W

img_result = zeros(img_w, img_l);
k = floor(N/2);
img_pad = padarray(imgdble, double([k k]), 0);
for i = double(1+k):img_w
    for j = double(1+k):img_l
        x = i - k;
        y = j - k;
        T = double(img_pad(i-k:i+k, j-k:j+k));
        img_result(x, y) = sum(sum(W.*T));
    end
end

%% Gradients respect to x

gx = zeros(img_w-2, img_l-2);
gy = zeros(img_w-2, img_l-2);

for i = 2:img_w-1
    for j = 2:img_l-1
        m = i - 1;
        n = j - 1;
        gx(m, n) = (img_result(i + 1, j) - img_result(i - 1, j))/2;
        gy(m, n) = (img_result(i, j + 1) - img_result(i, j - 1))/2;
    end
end

G = sqrt(gx.^2 + gy.^2);

%% Hyterisis

T1 = 0.10;
T2 = 0.07;

aboveT2 = G > T2;
[aboveT1row, aboveT1col] = find(img_result > T1);
edge_img = bwselect(aboveT2, aboveT1col, aboveT1row, 8);

%% Hough Transform for Line Detection

% Algorithm is mirrored from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

radius_step = 1
angle_step = 1

rho = 1:radius_step:hypot(size(edge_img,1), size(edge_img,2));
theta = 0:angle_step:180-angle_step;


%Voting procedures
hough_matrix = zeros(length(rho),length(theta)); % initialize the hough_matrix --- (p, theta) pairs

[y_idx, x_idx] = find(edge_img); % get x,y of edge pixels

for i = 1:numel(x_idx)
    hough_y = 0;
    for j = theta
        hough_y = hough_y + 1;
        r = x_idx(i) * cosd(j) + y_idx(i) * sind(j);
        if r >= 1 && r <= rho(end)
            temp = abs(r-rho);
            mintemp = min(temp);
            hough_x = find(temp == mintemp);
            
            % vote!
            hough_matrix(hough_x,hough_y) = hough_matrix(hough_x,hough_y)+1;
        end
    end
end

figure
imshow(imadjust(rescale(hough_matrix)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit');
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

threshold = 4;
numpeaks = 7;
[M, N] = size(hough_matrix);
peaks = zeros(numpeaks,2);
nHoodSize = floor(size(hough_matrix) / 100.0) * 2 + 1;

% Finding local maxima in hough_matrix
for m=1:numpeaks
        Hs = sort(hough_matrix(:),'descend');
        
        pkval = Hs(1);
        if pkval >= threshold
            [h,k] = find(hough_matrix==pkval,1);

            % Suppression
            lowX = max([floor(h-nHoodSize(1)) 1]);
            highX = min([ceil(h+nHoodSize(1)) M]);
            lowY = max([floor(k-nHoodSize(2)) 1]);
            highY = min([ceil(k+nHoodSize(2)) N]);
            hough_matrix(lowX:highX,lowY:highY) = 0;

            peaks(m,:) = [h, k];
        end

end

%% Get the lines

lines = houghlines(edge_img,theta,rho,peaks,'FillGap',5,'MinLength',20);
% lines = houghlines(edge_img,theta,p,[pdetect,thetadetect],'FillGap',5,'MinLength',20);
figure, imshow(edge_img), hold on
max_len = 0;
points_characteristics = [];
points = [];
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
   points = [points; xy];
end

p1 = transpose(points(find(points(:,1)==min(points(:,1))) , :));
p2 = transpose(points(find(points(:,1)==max(points(:,1))) , :));
p3 = transpose(points(find(points(:,2)==min(points(:,2))) , :));
p4 = transpose(points(find(points(:,2)==max(points(:,2))) , :));
% p4 = [172; 30];

c1 = [1; 1];
c2 = [1; size(edge_img,2)];
c3 = [size(edge_img,1); 1];
c4 = [size(edge_img,1); size(edge_img,2)];

P = [[p1; 1], [p2; 1], [p3; 1], [p4; 1],];

%% Image rectification

P_tmp = [{p1};{p2};{p3};{p4}];
C_tmp = [{c1};{c2};{c3};{c4}];
K = [];

f_dist = @(o1, o2) sqrt((o1(2) - o2(2))^2 + (o1(1) - o2(1))^2);

for i=1:4
    p = P_tmp{i};
    D = [f_dist(p, c1) f_dist(p, c2) f_dist(p, c3) f_dist(p, c4)];
    [~, idx] = min(D);
    K = [K, [C_tmp{idx};1]];
end

A = zeros(8,9);

for i=1:4
    l = i*2;
    A(l-1,:) = [-P(1,i) -P(2,i) -1 0 0 0 K(1,i)*P(1,i) K(1,i)*P(2,i) K(1,i)];
    A(l,:) = [0 0 0 -P(1,i) -P(2,i) -1 K(2,i)*P(1,i) K(2,i)*P(2,i) K(2,i)];
end

[U, S, V] = svd(A);

H = (reshape(V(:,9), 3, 3)).';

% Fun time!
rect_img = int8(zeros(size(edge_img,1), size(edge_img,2)));
[r_img_x, r_img_y] = find(rect_img==0);

rect_img_xy = [transpose(r_img_x); transpose(r_img_y); ones(1, length(r_img_x))];

old_img_loc = inv(H) * rect_img_xy;
p = old_img_loc(3,:);
old_img_loc = [old_img_loc(1,:)./p; old_img_loc(2,:)./p];
old_img_loc = floor(old_img_loc);

rect_img = uint8(zeros(size(edge_img,1), size(edge_img,2), 3));
k = 1;
for i=1:size(rect_img,2)
   for j=1:size(rect_img,1)
       rect_img(i,j,:) = img(old_img_loc(2,k), old_img_loc(1,k), :);
       k = k + 1;
   end
end

figure
imshow(rect_img);
