function [outimg] = bilinearInterpolation(im, scale)
% This function resize a given image with scaling factors of width and
% height.
% The algorithm used in this function is written after Richard Alan Peters' II digital image 
% processing slides on interpolation -- found at:
% https://ia902707.us.archive.org/23/items/Lectures_on_Image_Processing/EECE_4353_15_Resampling.pdf

    [org_row, org_col, num_chan] = size(im);
    mod_row = floor(scale(1)*org_row);
    mod_col = floor(scale(2)*org_col);
    outimg = zeros(mod_row, mod_col, num_chan);

    % Generate (x,y) pairs for each point in the output image
    % c_out = {[1, 2, 3,...,mod_col]
    %          [1, 2, 3,...,mod_col]
    %          ...
    %          [1, 2, 3,...,mod_col]}
    % r_out = {[1, 1, 1,...,1]
    %          [2, 2, 2,...,2]
    %          ...
    %          [mod_row, mod_row, mod_row,...,mod_row]}
    [c_out, r_out] = meshgrid(1 : mod_col, 1 : mod_row);

    % Scale these coordinates to match the one in the original picture
    r_out = r_out * (1/scale(1));
    c_out = c_out * (1/scale(2));

    % Since the coordinates can only be integer, we floor them and call this
    % the integral part of (r_out, c_out)
    r_0 = floor(r_out);
    c_0 = floor(c_out);

    % Cut out of range values. Any coordinate "0" will be 1 (matlab starting
    % index); Any coordinate > size of orginal image = last index of orginal
    % image. ==>> (r_0, c_0) are the row and column indices of the pixels in the
    % original image use to create the output image
    r_0(r_0 < 1) = 1;
    c_0(c_0 < 1) = 1;
    r_0(r_0 > org_row - 1) = org_row - 1;
    c_0(c_0 > org_col - 1) = org_col - 1;

    % (r_0, c_0) is the integral part, so delta_r and delta_c is is fractional part
    % of (r_out, c_out)
    delta_r = r_out - r_0;
    delta_c = c_out - c_0;



    % Then, the value of each output pixel is given by:
    % J(r_out, c_out) = I(r_0, c_0)*(1 - delta_r)*(1 - delta_c) +
    %                   I(r_0+1, c_0)*(delta_r)*(1 - delta_c) +
    %                   I(r_0, c_0+1)*(1 - delta_r)*(delta_c) +
    %                   I(r_0+1, c_0+1)*(delta_r)*(delta_c)
    for idx = 1:num_chan
        channel = im(:,:,idx); %// Get i'th channel
        %// Interpolate the channel
        tmp = zeros(mod_row, mod_col);
        for i = 1:mod_row
           for j = 1:mod_col
               m_0 = r_0(i,j);
               n_0 = c_0(i,j);
               tmp(i, j) = channel(m_0, n_0) * (1 - delta_r(i,j)) * (1 - delta_c(i,j)) + ...
                           channel(m_0+1, n_0) * (delta_r(i,j)) * (1 - delta_c(i,j)) + ...
                           channel(m_0, n_0+1) * (1 - delta_r(i,j)) * (delta_c(i,j)) + ...
                           channel(m_0+1, n_0+1) * (delta_r(i,j)) * (delta_c(i,j));
           end
        end
        outimg(:,:,idx) = tmp;
    end
end
    
    