function result = f_d(z)
    result = ( exp(-z) .* (z + 1) + 1 ) ./ ( ( 1 + exp(-z) ) .* ( 1 + exp(-z) ) );
    result(isnan(result)) = 0;
end