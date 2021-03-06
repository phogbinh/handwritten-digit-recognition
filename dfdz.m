function result = dfdz(z) % derivative of activation function
    result = ( exp(-z) .* (z + 1) + 1 ) ./ ( ( 1 + exp(-z) ) .* ( 1 + exp(-z) ) );
end