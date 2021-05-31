function result = f(x)
    result = x ./ (1 + exp(-x));
    result(isnan(result)) = 0;
end