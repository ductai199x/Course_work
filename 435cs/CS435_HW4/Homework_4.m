% Tai Duc Nguyen - CS 435 - Assignment 4 - 05/10/2019

%% Import all images

clear all;
close all;

root_path = fileparts(mfilename('fullpath'))
folder_path = [root_path, '/CarData/TrainImages/']
images_file = [root_path, '/images.dat']

% Check if data file exist, if so, import images data, if not, create file
if exist(images_file, 'file') == 2
% if false
     load(images_file, 'pos_images', 'neg_images', '-mat');
else
    files = dir([folder_path, '*.pgm']);
    L = length(files);
    pos_images = {};
    neg_images = {};
    n = 1; p = 1;

    for i=1:L
        fname = files(i).name;
        im = imread([folder_path, fname]);
%         hist = imhist(im);
%         hist = hist/sum(hist);
        
        if fname(1:3) == "pos"
            pos_images{p} = im;
            p = p + 1;
        else
            neg_images{n} = im;
            n = n + 1;
        end
    end

    save(images_file, 'pos_images', 'neg_images');
end

%% Set RNG seed

seed = 0
rng(seed);

%% Part 1: Classifying an Image using Grayscale Histograms


%% Get histograms

pos_hist = cellfun(@(x) imhist(x)/sum(imhist(x)), pos_images, 'UniformOutput', false);
neg_hist = cellfun(@(x) imhist(x)/sum(imhist(x)), neg_images, 'UniformOutput', false);

%% Get test set

pos_L = length(pos_hist);
neg_L = length(neg_hist);
test_pos = randperm(pos_L, floor(pos_L*1/3));
test_neg = randperm(neg_L, floor(neg_L*1/3));
test_set = [pos_hist{test_pos} neg_hist{test_neg}];
test_pos_index = length(test_pos);
test_neg_index = length(test_neg);

%% Get training set

train_pos = setdiff(1:pos_L, test_pos);
train_neg = setdiff(1:neg_L, test_neg);
train_set = [pos_hist{train_pos} neg_hist{train_neg}];
train_pos_index = length(train_pos);
train_neg_index = length(train_neg);


%% Define distance function

f_sim = @(w1,w2) sum(min([w1,w2],[],2));

%% Calculate accuracy of prediction

correct_1 = zeros(1, size(test_set, 2));

for i=1:size(test_set, 2)
   test_im = test_set(:,i);
   
   applyToGivenCol = @(func, test, train) @(col) func(test, train(:, col));
   applyToCol = @(func, test, train) arrayfun(applyToGivenCol(func, test, train), 1:size(train,2));

   dist_mat = applyToCol(f_sim, test_im, train_set);
   [~, max_idx] = max(dist_mat);
   predict_class = (max_idx <= train_pos_index);
   real_class = (i <= test_pos_index);
   if predict_class == real_class
       correct_1(i) = 1;
   end
end

fprintf("Classifying an Image using Grayscale Histograms\nAccuracy: %f\n", (sum(correct_1)/size(test_set, 2)));


%% Part 2: Part 2: Classifying an Image using Gists

%% Get 20x20 cells

cell_dim = [20 20];
im_dim = size(pos_images{1,1});
blockVectorR = [cell_dim(1) * ones(1, im_dim(1)/cell_dim(1))];
blockVectorC = [cell_dim(2) * ones(1, im_dim(2)/cell_dim(2))];
pos_cells = cellfun(@(x) mat2cell(x, blockVectorR, blockVectorC), pos_images, 'UniformOutput', false);
neg_cells = cellfun(@(x) mat2cell(x, blockVectorR, blockVectorC), neg_images, 'UniformOutput', false);

%% Get HOG 

bin_num = 8
base = 180/bin_num;
pos_HOG = cell(1, length(pos_cells));
neg_HOG = cell(1, length(neg_cells));

for i=1:length(pos_cells)
    im = pos_cells{i};
    
    HOG = [];
    for j=1:size(im,1)*size(im,2)
        C = im{j};
        [mag, dir] = imgradient(C, 'sobel');
        mag = reshape(mag,1,[]);    % normalize
        dir = abs(reshape(dir.',[],1)); % unsigned gradient
        
        dir = floor(dir/base)+1; dir(dir>bin_num) = 1; dir = [dir; bin_num];
        HOG = [HOG; accumarray(dir, [mag 0])];
    end
    pos_HOG{i} = HOG;
end

for i=1:length(neg_cells)
    im = neg_cells{i};
    
    HOG = [];
    for j=1:size(im,1)*size(im,2)
        C = im{j};
        [mag, dir] = imgradient(C, 'sobel');
        mag = reshape(mag,1,[]);    % normalize
        dir = abs(reshape(dir.',[],1)); % unsigned gradient
        
        dir = floor(dir/base)+1; dir(dir>bin_num) = 1; dir = [dir; bin_num];
        HOG = [HOG; accumarray(dir, [mag 0])];
    end
    neg_HOG{i} = HOG;
end

%% Get test set

test_set = [pos_HOG{test_pos} neg_HOG{test_neg}];

%% Get training set

train_set = [pos_HOG{train_pos} neg_HOG{train_neg}];

%% Calculate accuracy of prediction

correct_2 = zeros(1, size(test_set, 2));

for i=1:size(test_set, 2)
   test_im = test_set(:,i);
   
   applyToGivenCol = @(func, test, train) @(col) func(test, train(:, col));
   applyToCol = @(func, test, train) arrayfun(applyToGivenCol(func, test, train), 1:size(train,2));

   dist_mat = applyToCol(f_sim, test_im, train_set);
   [~, max_idx] = max(dist_mat);
   predict_class = (max_idx <= train_pos_index);
   real_class = (i <= test_pos_index);
   if predict_class == real_class
       correct_2(i) = 1;
   end
end

fprintf("Classifying an Image using Gist\nAccuracy: %f\n", (sum(correct_2)/size(test_set, 2)));










