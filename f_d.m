function result = f_d(x)
    result = ( exp(-x) .* (x + 1) + 1 ) ./ ( ( 1 + exp(-x) ) .* ( 1 + exp(-x) ) );
    result(isnan(result)) = 0;
end