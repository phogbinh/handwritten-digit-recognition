function result = df_dz(z)
    result = ( exp(-z) .* (z + 1) + 1 ) ./ ( ( 1 + exp(-z) ) .* ( 1 + exp(-z) ) );
end