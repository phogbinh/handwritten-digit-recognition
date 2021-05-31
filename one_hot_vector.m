function result = one_hot_vector(digit)
    result = zeros(10, 1);
    result(digit + 1) = 1;
end