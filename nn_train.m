TRAIN_DATA = load('mnist_train.csv');
TRAIN_DATA_N = numel( TRAIN_DATA(:, 1) );
TRAIN_IN = normalized_grayscale( TRAIN_DATA(:, 2:785) );
TRAIN_OU = TRAIN_DATA(:, 1);
clear TRAIN_DATA;
LAYERS_NEURONS_N = readmatrix('architect'); % [N]umber of [NEURONS] in [L]ayers
LAYERS_N = numel( LAYERS_NEURONS_N(:, 1) ); % [N]umber of [L]ayers
clear LAYERS_NEURONS_N;
L(LAYERS_N, 1) = layer;

for l_i = 2:LAYERS_N
    L(l_i).b = readmatrix( strcat( 'b', num2str(l_i) ) );
    L(l_i).w = readmatrix( strcat( 'w', num2str(l_i) ) );
end

for train_round_i = 1:nn.TRAIN_ROUNDS_N
    for mini_batch_i = 1:(TRAIN_DATA_N / nn.MINI_BATCH_LENGTH)
        for l_i = 2:LAYERS_N
            L(l_i).dcdb_cum = zeros( size( L(l_i).b ) );
            L(l_i).dcdw_cum = zeros( size( L(l_i).w ) );
        end

        start_i = (mini_batch_i - 1) * nn.MINI_BATCH_LENGTH + 1;
        end_i = start_i + nn.MINI_BATCH_LENGTH - 1;
        for train_data_i = start_i:end_i
            desired_output_layer = one_hot_vector( TRAIN_OU(train_data_i) );

            % feedfoward
            L(1).y = transpose( TRAIN_IN(train_data_i, :) );
            for l_i = 2:LAYERS_N
                L(l_i).z = L(l_i).w * L(l_i-1).y + L(l_i).b;
                L(l_i).y = f( L(l_i).z );
            end

            % backpropagation
            dcdb = ( 2 * ( L(LAYERS_N).y - desired_output_layer ) ) .* df_dz( L(LAYERS_N).z );
            dcdw = dcdb * transpose( L(LAYERS_N - 1).y );
            L(LAYERS_N).dcdb_cum = L(LAYERS_N).dcdb_cum + dcdb;
            L(LAYERS_N).dcdw_cum = L(LAYERS_N).dcdw_cum + dcdw;
            for l_i = (LAYERS_N - 1):-1:2
                dcdb = ( transpose( L(l_i + 1).w ) * dcdb ) .* df_dz( L(l_i).z );
                dcdw = dcdb * transpose( L(l_i - 1).y );
                L(l_i).dcdb_cum = L(l_i).dcdb_cum + dcdb;
                L(l_i).dcdw_cum = L(l_i).dcdw_cum + dcdw;
            end
        end

        for l_i = 2:LAYERS_N
            L(l_i).b = L(l_i).b - L(l_i).dcdb_cum ./ nn.MINI_BATCH_LENGTH * nn.LEARNING_RATE;
            L(l_i).w = L(l_i).w - L(l_i).dcdw_cum ./ nn.MINI_BATCH_LENGTH * nn.LEARNING_RATE;
        end
    end
end

save('trained_parameters', 'L');
clear;