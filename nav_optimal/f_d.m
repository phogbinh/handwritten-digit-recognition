function y = f_d(x)
    y = ( exp(-x) .* (x + 1) + 1 ) ./ ( ( 1 + exp(-x) ) .* ( 1 + exp(-x) ) );
    y(isnan(y)) = 0;
    y(isinf(y)) = 1;
end