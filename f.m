function y = f(x)
    y = x ./ (1 + exp(-x));
    y(isnan(y)) = 0;
end