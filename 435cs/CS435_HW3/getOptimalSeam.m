function M = getOptimalSeam(E, M, i)
% E is energy function
% M is current seam matrix
% i is the current row being processed

    if i == 1
        M(1,:) = E(1,:);
    else 
        for j=1:size(M,2)
            if j == 1
                M(i,j) = E(i,j) + min([M(i-1, j), M(i-1, j+1)]);
            elseif j == size(M,2)
                M(i,j) = E(i,j) + min([M(i-1, j-1), M(i-1, j)]);
            else
                M(i,j) = E(i,j) + min([M(i-1, j-1), M(i-1, j), M(i-1, j+1)]);
            end
        end
    end
end