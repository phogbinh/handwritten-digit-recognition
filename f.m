function result = f(z)
    result = z ./ (1 + exp(-z));
    result(isnan(result)) = 0;
end