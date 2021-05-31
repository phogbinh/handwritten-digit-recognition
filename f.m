function result = f(z) % activation function
    result = z ./ (1 + exp(-z));
    result(isnan(result)) = 0;
end