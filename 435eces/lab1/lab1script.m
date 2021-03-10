% lab1script

% use the 'imread' command to read in an image
pout= imread('pout.tif');

% If the image is black and white, this will create a two dimensional array
% of all of the image's pixel values.

% Note: when an image is read in to Matlab, it is of datatyp e uint8 (i.e.
% unsigned 8 bit integer)

% use the size command to find the dimensions of the image
[poutRows,poutCols]= size(pout)

pause
%%%%%%%%%%%%%%%%%%%%


% use the 'imshow' command to display the image
figure
imshow(pout)

pause
%%%%%%%%%%%%%%%%%%%%

% Note: In order to use the imshow command, the image to display must be of
% datatype 'uint8'.  This command will display an all black or white image
% if the image is another datatype such as 'double'.  If you use any Matlab 
% tools to process an image, it is likely that the output will be a 
% 'double' instead of a 'unit8'.

% use the 'double()' command to cast a variable to data type 'double'
% use the 'uint8()' command to cast a variable to data type 'uint8'
% use the 'class' command to determine the data type of a variable

poutDouble= double(pout);
figure
imshow(poutDouble)

class(poutDouble)
pause

imshow(uint8(poutDouble))
pause
%%%%%%%%%%%%%%%%%%%%


% if you have too many image windows open, you can use the 'close all'
% command to close all open Matlab figures

close all
pause
%%%%%%%%%%%%%%%%%%%%


% To examine a grayscale image's pixel value histogram, use the imhist
% command.

% Note: For imhist to work, the image must be of data type 'uint8'

imhist(pout)

pause

% you can also use the 'imhist' command to return the histogram bin
% locations and values

[hval,pixval]= imhist(pout);

% you can then create a bar graph using these values (or use them for image
% analysis/processing)

figure
bar(pixval,hval)

pause
close all
%%%%%%%%%%%%%%%%%%%%


% Use the 'histeq' command to perform histogram equalization on an image
poutHisEq= histeq(pout);

figure
imshow(poutHisEq)
pause

figure
imhist(poutHisEq)
pause

% When performing histogram equalization, the 'histeq' command uses 64
% histogram bins by default.  However, you can specify the number of
% histogram bins that the function uses as follows:
poutHisEq2= histeq(pout,256);

figure
imshow(poutHisEq2)
pause

figure
imhist(poutHisEq2)
pause
close all
%%%%%%%%%%%%%%%%%%%%


% read in a new image
peppers= imread('peppers.tif');

figure
imshow(peppers)
pause

% To filter an image, use the 'filter2' command.

avgMask5= ones(5)/25; % create a 5x5 averaging filter

peppersAvg5= filter2(avgMask5,peppers); 

% Note: The first argument should be the filter and the second argument
% should be the image.  Additionally, this command will return an output of
% type 'double'.  The image must be cast to a 'uint8' before it can be
% displayed.

figure
imshow(uint8(peppersAvg5))
pause
close all
%%%%%%%%%%%%%%%%%%%%


% To add noise to an image, use the 'rand' function to generate uniformly
% distributed noise or the 'randn' function to generate Gaussian
% distributed noise

noisevar= 128;
peppersnoise= uint8( double(peppers) + sqrt(noisevar)*randn(512,512));

% display original peppers image
figure
imshow(peppers)
pause

% display noisy peppers image
figure
imshow(peppersnoise)
pause

% use a 3x3 averaging filter to densoise the image
avgMask3= ones(3)/9;
peppersAvg3= filter2(avgMask3,peppersnoise);
figure
imshow(uint8(peppersAvg3))
pause

% use ta 3x3 median filter to denoise the image
peppersMed3= medfilt2(peppersnoise,[3 3]);
figure
imshow(peppersMed3)
pause
close all
%%%%%%%%%%%%%%%%%%%%


% Sobel filter for row direction
SobelX= [
1 0 -1
2 0 -2
1 0 -1 ];

% Sobel filter for column directon
SobelY= SobelX.';

% calculate the gradient in each cardinal direction
Gx= filter2(SobelX,peppers); % x-direction
Gy= filter2(SobelY,peppers); % y-direction

% calculate the magnitude of the gradient
gradMag= (Gx.^2 + Gy.^2).^.5;

% show the magnitude of the image's gradient
figure
imshow(uint8(gradMag))
pause

% calculate an edge map of the image by comparing the magnitude of the
% gradient to a threshold
threshold= 128;
edgemap= gradMag > threshold;

figure
imshow(edgemap)

pause
close all
%%%%%%%%%%%%%%%%%%%%



% read in a color image
baboon= imread('baboon_color.tiff');

% use the size command to find the dimensions of the image
[babRows,babCols,babColors]= size(baboon)

% display the color image
figure
imshow(baboon);
pause

% get each color layer of the image
baboonRed= baboon(:,:,1); % red color layer
baboonGreen= baboon(:,:,2); % green color layer
baboonBlue= baboon(:,:,3); % blue color layer

% display each color layer
figure
imshow(baboonRed)
pause

figure
imshow(baboonGreen)
pause

figure
imshow(baboonBlue)
pause
close all

% convert the image to the Y Cb Cr color space (Y is luminance, Cb and Cr
% are chrominance)
baboonYCbCr= rgb2ycbcr(baboon);

% display the luminance color layer of baboon image
figure
imshow(baboonYCbCr(:,:,1))
pause

% display the chrominance color layers of the baboon image
figure
imshow(baboonYCbCr(:,:,2))
pause

figure
imshow(baboonYCbCr(:,:,3))
