function P = backtrack(M)
% M is the seam matrix
% P is the optimal path
prev = 0;
P = zeros(size(M,1), size(M,2));
for i=size(M,1):-1:1
    if i == size(M,1)
        [~, idx] = min(M(i,:));
        prev = idx;
    else
        if prev == 1
            [~, idx] = min(M(i,prev:prev+1));
        elseif prev == size(M,2)
            [~, idx] = min(M(i,prev-1:prev));
        else
            [~, idx] = min(M(i,prev-1:prev+1));
        end
        prev = idx - 2 + prev;
    end
    P(i, prev) = 1;
end