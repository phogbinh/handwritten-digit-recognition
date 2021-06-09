%% input
TRAIN_DATA = load('mnist_train.csv');
TRAIN_DATA_N = numel( TRAIN_DATA(:, 1) );
TRAIN_IN = normalized_grayscale( TRAIN_DATA(:, 2:785) );
TRAIN_OU = TRAIN_DATA(:, 1);
clear TRAIN_DATA;
% b's
b2 = readmatrix('b2');
b3 = readmatrix('b3');
b4 = readmatrix('b4');
% w's
w2 = readmatrix('w2');
w3 = readmatrix('w3');
w4 = readmatrix('w4');
MINI_BATCH_LEARNING_RATE = nn.LEARNING_RATE / nn.MINI_BATCH_LENGTH;

%% train
tic;
for train_round_i = 1:nn.TRAIN_ROUNDS_N
    for mini_batch_i = 1:(TRAIN_DATA_N / nn.MINI_BATCH_LENGTH)
        % initialize -- parallelable
        start_i = (mini_batch_i - 1) * nn.MINI_BATCH_LENGTH + 1;
        end_i = start_i + nn.MINI_BATCH_LENGTH - 1;

        PAGES_N = end_i - start_i + 1;
        DCDB2 = zeros( [47 1 PAGES_N] );
        DCDB3 = zeros( [53 1 PAGES_N] );
        DCDB4 = zeros( [10 1 PAGES_N] );
        DCDW2 = zeros( [47 784 PAGES_N] );
        DCDW3 = zeros( [53 47 PAGES_N] );
        DCDW4 = zeros( [10 53 PAGES_N] );
        
        train_in = TRAIN_IN(start_i:end_i, :);
        train_ou = TRAIN_OU(start_i:end_i);
        
        parfor page_i = 1:PAGES_N
            desired_output_layer = one_hot_vector( train_ou(page_i) );

            % feedforward -- sequential required
            y1 = transpose( train_in(page_i, :) );
            z2 = w2 * y1 + b2;
            y2 = f( z2 );
            z3 = w3 * y2 + b3;
            y3 = f( z3 );
            z4 = w4 * y3 + b4;
            y4 = f( z4 );

            % backpropagation -- sequential required
            dcdb = ( 2 * ( y4 - desired_output_layer ) ) .* dfdz( z4 );
            DCDB4(:, :, page_i) = dcdb;
            DCDW4(:, :, page_i) = dcdb * transpose( y3 );
            
            dcdb = ( transpose( w4 ) * dcdb ) .* dfdz( z3 );
            DCDB3(:, :, page_i) = dcdb;
            DCDW3(:, :, page_i) = dcdb * transpose( y2 );
            
            dcdb = ( transpose( w3 ) * dcdb ) .* dfdz( z2 );
            DCDB2(:, :, page_i) = dcdb;
            DCDW2(:, :, page_i) = dcdb * transpose( y1 );
        end

        % update -- parallelable
        b2 = b2 - sum(DCDB2, 3) * MINI_BATCH_LEARNING_RATE;
        b3 = b3 - sum(DCDB3, 3) * MINI_BATCH_LEARNING_RATE;
        b4 = b4 - sum(DCDB4, 3) * MINI_BATCH_LEARNING_RATE;
        w2 = w2 - sum(DCDW2, 3) * MINI_BATCH_LEARNING_RATE;
        w3 = w3 - sum(DCDW3, 3) * MINI_BATCH_LEARNING_RATE;
        w4 = w4 - sum(DCDW4, 3) * MINI_BATCH_LEARNING_RATE;
    end
end
t = toc;
disp( ['cpu time: ' num2str(t)] );

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