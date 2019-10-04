%% Tai Duc Nguyen - Problem set 1

%% Problem 1

% When given a 2-dimensional pdf, the graph of the pdf is 3-dimensional.
% Hence, when provided point [0, 1], which is exactly at the mean, hence z
% will the the heighest value of the graph (z direction), a.k.a the most
% likely probability. Therefore, max(p) is 1.5586


%% Problem 2

mu = [0,1]
sigma = [0.1 0; 0 0.1]


[X1,X2] = meshgrid(linspace(-5,5,500)',linspace(-5,5,500)');
X = [X1(:) X2(:)];
p = mvnpdf(X, mu, sigma);
surf(X1,X2,reshape(p,500,500));
max(p)
