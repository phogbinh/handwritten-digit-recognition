%% input
TRAIN_DATA = load('mnist_train.csv');
TRAIN_DATA_N = numel( TRAIN_DATA(:, 1) );
TRAIN_IN = normalized_grayscale( TRAIN_DATA(:, 2:785) );
TRAIN_OU = TRAIN_DATA(:, 1);
clear TRAIN_DATA;
LAYERS_NEURONS_N = readmatrix('architect');
LAYERS_N = numel( LAYERS_NEURONS_N(:, 1) );
clear LAYERS_NEURONS_N;
% b's
b2 = readmatrix('b2');
b3 = readmatrix('b3');
b4 = readmatrix('b4');
% w's
w2 = readmatrix('w2');
w3 = readmatrix('w3');
w4 = readmatrix('w4');
layers_associates(LAYERS_N, 1) = layer_associates;
MINI_BATCH_LEARNING_RATE = nn.LEARNING_RATE / nn.MINI_BATCH_LENGTH;

%% train
tic;
for train_round_i = 1:nn.TRAIN_ROUNDS_N
    for mini_batch_i = 1:(TRAIN_DATA_N / nn.MINI_BATCH_LENGTH)
        % initialize
        layers_associates(2).dcdb_cum = zeros( size( b2 ) );
        layers_associates(3).dcdb_cum = zeros( size( b3 ) );
        layers_associates(4).dcdb_cum = zeros( size( b4 ) );
        layers_associates(2).dcdw_cum = zeros( size( w2 ) );
        layers_associates(3).dcdw_cum = zeros( size( w3 ) );
        layers_associates(4).dcdw_cum = zeros( size( w4 ) );

        start_i = (mini_batch_i - 1) * nn.MINI_BATCH_LENGTH + 1;
        end_i = start_i + nn.MINI_BATCH_LENGTH - 1;
        for train_data_i = start_i:end_i
            desired_output_layer = one_hot_vector( TRAIN_OU(train_data_i) );

            % feedforward
            layers_associates(1).y = transpose( TRAIN_IN(train_data_i, :) );
            layers_associates(2).z = w2 * layers_associates(1).y + b2;
            layers_associates(2).y = f( layers_associates(2).z );
            layers_associates(3).z = w3 * layers_associates(2).y + b3;
            layers_associates(3).y = f( layers_associates(3).z );
            layers_associates(4).z = w4 * layers_associates(3).y + b4;
            layers_associates(4).y = f( layers_associates(4).z );

            % backpropagation
            dcdb = ( 2 * ( layers_associates(4).y - desired_output_layer ) ) .* dfdz( layers_associates(4).z );
            dcdw = dcdb * transpose( layers_associates(3).y );
            layers_associates(4).dcdb_cum = layers_associates(4).dcdb_cum + dcdb;
            layers_associates(4).dcdw_cum = layers_associates(4).dcdw_cum + dcdw;
            
            dcdb = ( transpose( w4 ) * dcdb ) .* dfdz( layers_associates(3).z );
            dcdw = dcdb * transpose( layers_associates(2).y );
            layers_associates(3).dcdb_cum = layers_associates(3).dcdb_cum + dcdb;
            layers_associates(3).dcdw_cum = layers_associates(3).dcdw_cum + dcdw;
            
            dcdb = ( transpose( w3 ) * dcdb ) .* dfdz( layers_associates(2).z );
            dcdw = dcdb * transpose( layers_associates(1).y );
            layers_associates(2).dcdb_cum = layers_associates(2).dcdb_cum + dcdb;
            layers_associates(2).dcdw_cum = layers_associates(2).dcdw_cum + dcdw;
        end

        % update
        b2 = b2 - layers_associates(2).dcdb_cum * MINI_BATCH_LEARNING_RATE;
        b3 = b3 - layers_associates(3).dcdb_cum * MINI_BATCH_LEARNING_RATE;
        b4 = b4 - layers_associates(4).dcdb_cum * MINI_BATCH_LEARNING_RATE;
        w2 = w2 - layers_associates(2).dcdw_cum * MINI_BATCH_LEARNING_RATE;
        w3 = w3 - layers_associates(3).dcdw_cum * MINI_BATCH_LEARNING_RATE;
        w4 = w4 - layers_associates(4).dcdw_cum * MINI_BATCH_LEARNING_RATE;
    end
end
t = toc;
disp( ['cpu time: ' num2str(t)] );

%% output
layers(LAYERS_N, 1) = layer;
% b's
layers(2).b = b2;
layers(3).b = b3;
layers(4).b = b4;
% w's
layers(2).w = w2;
layers(3).w = w3;
layers(4).w = w4;
% save
save('trained_parameters', 'layers');
clear;