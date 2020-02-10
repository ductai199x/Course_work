%% Tai Duc Nguyen, Hieu Mai - 02/01/2020 - ECES 435

clear all; close all;

%% PART 1

peppers_org = imread("peppers.tif");
baboon_org = imread("baboon.tif");

%%

fprintf('| %10s | %10s | %10s | %10s |\n', "Images", "Quality", "Size", "PSNR");
fprintf('-----------------------------------------------------\n');

quality_factors = [90, 70, 50, 30, 10];

for i = 1:length(quality_factors)
    filename = strcat('peppers_', num2str(quality_factors(i)), '.jpg');
    imwrite(peppers_org, filename, 'Quality', quality_factors(i));
    tmp_img = imread(filename);
    PSNR = 20*log10(255) - 10*log10(mse(tmp_img, peppers_org));
    fprintf('| %10s | %10d | %10lu | %10f |\n', "peppers", quality_factors(i), imfinfo(filename).FileSize, PSNR);
end
    
for i = 1:length(quality_factors)
    filename = strcat('baboon_', num2str(quality_factors(i)), '.jpg');
    imwrite(baboon_org, filename, 'Quality', quality_factors(i));
    tmp_img = imread(filename);
    PSNR = 20*log10(255) - 10*log10(mse(tmp_img, baboon_org));
    fprintf('| %10s | %10d | %10lu | %10f |\n', "baboon", quality_factors(i), imfinfo(filename).FileSize, PSNR);
end

fprintf("\r\n");


%% PART 2

current_dir = strcat(mfilename('fullpath'), '.m');
[current_dir,~,~] = fileparts(current_dir);

lum_quant = ...
[ 16 11 10 16 24 40 51 61;
  12 12 14 19 26 58 60 55;
  14 13 16 24 40 57 69 56;
  14 17 22 29 51 87 80 62;
  18 22 37 56 68 109 103 77;
  24 35 55 64 81 104 113 92;
  49 64 78 87 103 121 120 101;
  72 92 95 98 112 100 103 99;];

% lum_quant = ...
% [ 16 11 10 16 24 40 51 61;
%   12 12 14 19 26 58 60 55;
%   14 13 16 24 40 57 69 56;
%   14 17 22 29 51 1000 1000 62;
%   18 22 37 56 68 1000 1000 77;
%   24 35 55 64 1000 1000 1000 1000;
%   59 64 78 1000 1000 1000 1000 1000;
%   72 1000 1000 1000 1000 1000 1000 1000;];

[zz_quant_dct_blks, enc_size]= JPEG_encode(peppers_org, current_dir, lum_quant);
% [zz_quant_dct_blks, enc_size]= JPEG_encode(baboon_org, current_dir, lum_quant);


%%

current_dir = '/mnt/linux-data/2-coursework/435eces/assgn2';
[iZZDCTQIm, dec_img] = JPEG_decode(current_dir);

figure(1)
imshow(dec_img, []);

PSNR = 20*log10(255) - 10*log10(mse(dec_img, double(peppers_org)));


%%

function [zz_quant_dct_blks, enc_size] = JPEG_encode(X, current_dir, lum_quant)

    [nrow, ncol] = size(X);
    image_flat = X(:);

    block_size = 8;
    blocks = zeros(int16(length(image_flat)/(block_size^2)), block_size^2);
    k = 1;
    i = 1;
    while i <= length(image_flat)
        for j = 0:block_size-1
            m = i + ncol*j;
            n = j*block_size + 1;
            blocks(k,n:n+block_size-1) = image_flat(m:m+block_size-1);
        end

        if (mod(k,int16(ncol/block_size)))
            i = i + block_size;
        else
            i = block_size^2*k + 1;
        end
        k = k + 1;
    end

    zz_quant_dct_blks = zeros(size(blocks));

    for i = 1:size(zz_quant_dct_blks,1)
        blk_dct = dct2(reshape(blocks(i,:), block_size, block_size))./lum_quant;
%         blk_dct = dct2(reshape(blocks(i,:), block_size, block_size));
        zz_quant_dct_blks(i,:) = ZigzagMtx2Vector(blk_dct);
    end

    enc_size = JPEG_entropy_encode(nrow, ncol, block_size, ...
                        lum_quant, zz_quant_dct_blks, current_dir, []);

end


function [iZZDCTQIm, dec_img] = JPEG_decode(current_dir)
    [nrow,ncol,dct_block_size,iQ,iZZDCTQIm] = JPEG_entropy_decode(current_dir);
     
    dec_img = zeros(nrow, ncol);

    k = 1;
    for i = 1:dct_block_size:nrow
        for j = 1:dct_block_size:ncol
            dec_img(j:j+dct_block_size-1, i:i+dct_block_size-1) = ...
                idct2(iQ.*Vector2ZigzagMtx(iZZDCTQIm(k,:)));
            k = k + 1;
        end
    end

end



