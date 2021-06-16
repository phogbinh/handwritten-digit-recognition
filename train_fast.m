%% input
TRAIN_DATA = load('mnist_train.csv');
TRAIN_IN = gpuArray( normalized_grayscale( TRAIN_DATA(:, 2:785) ) );
TRAIN_OU = gpuArray( TRAIN_DATA(:, 1) );
clear TRAIN_DATA;
% b's
b2 = gpuArray( readmatrix('b2') );
b3 = gpuArray( readmatrix('b3') );
b4 = gpuArray( readmatrix('b4') );
% w's
w2 = gpuArray( readmatrix('w2') );
w3 = gpuArray( readmatrix('w3') );
w4 = gpuArray( readmatrix('w4') );

%% train
tic;

% initialize -- parallelable
dcdb_cum2 = zeros( size(b2) );
dcdb_cum3 = zeros( size(b3) );
dcdb_cum4 = zeros( size(b4) );
dcdw_cum2 = zeros( size(w2) );
dcdw_cum3 = zeros( size(w3) );
dcdw_cum4 = zeros( size(w4) );

% params update on each iteration -- sequential required
for train_round_i = 1:50
    % params update on each iteration -- sequential required
    for mini_batch_i = 1:6000
        % initialize -- parallelable
        dcdb_cum2(:) = 0;
        dcdb_cum3(:) = 0;
        dcdb_cum4(:) = 0;
        dcdw_cum2(:) = 0;
        dcdw_cum3(:) = 0;
        dcdw_cum4(:) = 0;

        start_i = (mini_batch_i - 1) * 10 + 1;
        end_i = start_i + 10 - 1;
        
        % gradient cumulation on each iteration -- parallelable with race
        % condition
        for train_data_i = start_i:end_i
            desired_output_layer = one_hot_vector( TRAIN_OU(train_data_i) );

            % feedforward -- sequential required
            y1 = transpose( TRAIN_IN(train_data_i, :) );
            z2 = w2 * y1 + b2;
            y2 = f( z2 );
            z3 = w3 * y2 + b3;
            y3 = f( z3 );
            z4 = w4 * y3 + b4;
            y4 = f( z4 );

            % backpropagation -- sequential required
            dcdb = ( 2 * ( y4 - desired_output_layer ) ) .* dfdz( z4 );
            dcdw = dcdb * transpose( y3 );
            dcdb_cum4 = dcdb_cum4 + dcdb;
            dcdw_cum4 = dcdw_cum4 + dcdw;
            
            dcdb = ( transpose( w4 ) * dcdb ) .* dfdz( z3 );
            dcdw = dcdb * transpose( y2 );
            dcdb_cum3 = dcdb_cum3 + dcdb;
            dcdw_cum3 = dcdw_cum3 + dcdw;
            
            dcdb = ( transpose( w3 ) * dcdb ) .* dfdz( z2 );
            dcdw = dcdb * transpose( y1 );
            dcdb_cum2 = dcdb_cum2 + dcdb;
            dcdw_cum2 = dcdw_cum2 + dcdw;
        end

        % update -- parallelable
        b2 = b2 - dcdb_cum2 * 0.0006;
        b3 = b3 - dcdb_cum3 * 0.0006;
        b4 = b4 - dcdb_cum4 * 0.0006;
        w2 = w2 - dcdw_cum2 * 0.0006;
        w3 = w3 - dcdw_cum3 * 0.0006;
        w4 = w4 - dcdw_cum4 * 0.0006;
    end
end

t = toc;
disp( ['computational time: ' num2str(t)] );

%% output
layers(4, 1) = layer;
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