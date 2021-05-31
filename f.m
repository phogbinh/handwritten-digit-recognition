function result = f(z) % activation function
    result = z ./ (1 + exp(-z)); % swish function
end