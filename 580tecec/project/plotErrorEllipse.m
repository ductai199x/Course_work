function plotErrorEllipse(mu, sigma, rho, color)
    s = -2 * log(1 - rho);
    [V, D] = eig(sigma * s);
    t = linspace(0, 2 * pi);
    a = (V * sqrt(D)) * [cos(t(:))'; sin(t(:))'];
    plot(a(1, :) + mu(1), a(2, :) + mu(2), color);
end
