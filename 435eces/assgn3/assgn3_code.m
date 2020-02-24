%% Tai Duc Nguyen, Hieu Mai - ECECS 435 - 02/20/2020

close all; clear all;

%% Part 1.1: LSB

peppers_org = imread('peppers.tif');
baboon_org = imread('baboon.tif');

figure(1)

subplot(2,2,1);
peppers_bitplane = ExtractBitPlane(peppers_org, 3);
imshow(peppers_bitplane, [0 1]);
title("3rd bit plane of peppers.tif");

subplot(2,2,2);
peppers_bitplane = ExtractBitPlane(peppers_org, 2);
imshow(peppers_bitplane, [0 1]);
title("2nd bit plane of peppers.tif");

subplot(2,2,3);
baboon_bitplane = ExtractBitPlane(baboon_org, 5);
imshow(baboon_bitplane, [0 1]);
title("5th bit plane of baboon.tif");

subplot(2,2,4);
baboon_bitplane = ExtractBitPlane(baboon_org, 4);
imshow(baboon_bitplane, [0 1]);
title("4th bit plane of baboon.tif");

%% Part 1.2: LSB watermark

LSBwmk1 = imread("LSBwmk1.tiff");
LSBwmk2 = imread("LSBwmk2.tiff");
LSBwmk3 = imread("LSBwmk3.tiff");


figure(2)

subplot(3,2,1)
imshow(LSBwmk1);
title("LSBwmk1.tiff");

subplot(3,2,2)
LSBwmk1_bitplane = ExtractBitPlane(LSBwmk1, 2);
imshow(LSBwmk1_bitplane, [0 1]);
title("Watermark in LSBwmk1.tiff");

subplot(3,2,3)
imshow(LSBwmk2);
title("LSBwmk2.tiff");

subplot(3,2,4)
LSBwmk2_bitplane = ExtractBitPlane(LSBwmk2, 1);
imshow(LSBwmk2_bitplane, [0 1]);
title("Watermark in LSBwmk2.tiff");

subplot(3,2,5)
imshow(LSBwmk3);
title("LSBwmk3.tiff");

subplot(3,2,6)
LSBwmk3_bitplane = ExtractBitPlane(LSBwmk3, 1);
imshow(LSBwmk3_bitplane, [0 1]);
title("Watermark in LSBwmk3.tiff");


%% Part 1.3: Embed Watermark

barbara_org = imread("Barbara.bmp");

figure(3)

subplot(2,2,1)
emb = LSBEmbedWatermark(peppers_org, barbara_org, 4);
imshow(emb);
title("Distortion appears at N=4");

subplot(2,2,2)
emb = LSBEmbedWatermark(peppers_org, barbara_org, 6);
imshow(emb);
title("Watermark's content appears at N=6");

subplot(2,2,3)
emb = LSBEmbedWatermark(baboon_org, barbara_org, 4);
imshow(emb);
title("Distortion appears at N=5");

subplot(2,2,4)
emb = LSBEmbedWatermark(baboon_org, barbara_org, 6);
imshow(emb);
title("Watermark's content appears at N=6");

%% Part 2.1: Yang-Mintzer

seed = 2020;
figure(4)

subplot(2,2,1)
peppers_emb = YMEmbedWatermark(peppers_org, barbara_org, seed);
psnr_ = psnr(peppers_emb, peppers_org);
imshow(peppers_emb);
title(["PSNR_{after}=", num2str(psnr_)]);

subplot(2,2,2)
bp = ExtractBitPlane(peppers_emb, 1);
imshow(bp, [0,1]);
title("Extracted LSB after YM");

subplot(2,2,3)
baboon_emb = YMEmbedWatermark(baboon_org, barbara_org, seed);
psnr_ = psnr(baboon_emb, baboon_org);
imshow(baboon_emb);
title(["PSNR_{after}=", num2str(psnr_)]);

subplot(2,2,4)
bp = ExtractBitPlane(baboon_emb, 1);
imshow(bp, [0,1]);
title("Extracted LSB after YM");

%% Part 2.2: YM Extract

YMembedded = imread("YMwmkedKey435.tiff");

figure(5)

subplot(1,2,1)
imshow(YMembedded);
title("YM embedded picture");

subplot(1,2,2)
wtm = YMExtractWatermark(YMembedded, 435);
imshow(wtm, [0,1]);
title("Extracted watermark");


%%
function bp = ExtractBitPlane(image, n)
    bp = zeros(size(image));
    tmp = double(image);
    max_bp = 8;
    for i=max_bp-1:-1:n-1
        bp = floor(tmp/(2^i));
        tmp = tmp - (2^i)*bp;
    end
    bp = uint8(bp);

end

function emb = LSBEmbedWatermark(host, wtm, n)
    max_bp = 8;
    bp1 = zeros([size(host) max_bp]);
    bp2 = zeros([size(wtm) max_bp]);
    
    host_tmp = double(host);
    wtm_tmp = double(wtm);
    for i=max_bp-1:-1:0
        bp1(:,:,i+1) = floor(host_tmp/(2^i));
        host_tmp = host_tmp - (2^i)*bp1(:,:,i+1);
        bp2(:,:,i+1) = floor(wtm_tmp/(2^i));
        wtm_tmp = wtm_tmp - (2^i)*bp2(:,:,i+1);
    end
    
    bp1(:,:,n:-1:1) = bp2(:,:,max_bp:-1:max_bp-n+1);
    emb = zeros(size(host));
    for i=0:1:max_bp-1
        emb = emb + bp1(:,:,i+1)*(2^i);
    end
    
    emb = uint8(emb);

end

function emb = YMEmbedWatermark(host, wtm, seed)
    max_bp = 8;
    wtm_tmp = wtm > 2^(max_bp-1);
    emb = zeros(size(host));
    rng(seed);
    LUTvals = rand(256) > 0.5;
    
    for i=1:size(emb, 1)
        for j=1:size(emb,2)
            pix = host(i,j);
            if(LUTvals(pix + 1) ~= wtm_tmp(i,j))
                while(LUTvals(pix + 1) ~= wtm_tmp(i,j))
                    pix = pix + 1;
                    if(pix >= 256)
                        pix = 1;
                    end
                end
                emb(i,j) = pix;
            else
                emb(i,j) = host(i,j);
            end
        end
    end
    
    emb = uint8(emb);

end

function wtm = YMExtractWatermark(img, seed)
    rng(seed);
    LUTvals = rand(size(img)) > 0.5;
    wtm = zeros(size(img));
    for i=1:size(img, 1)
        for j=1:size(img,2)
            wtm(i,j) = LUTvals(img(i,j) + 1);
        end
    end

end