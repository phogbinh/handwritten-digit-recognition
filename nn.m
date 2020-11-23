train_data = load('mnist_train.csv');
train_data_n = numel( train_data(:, 1) );
train_in = train_data(:, 2:785);
train_ou = train_data(:, 1);

L_NEURONS_N = readmatrix('architect'); % [N]umber of [NEURONS] in [L]ayers
L_N = numel( L_NEURONS_N(:, 1) ); % [N]umber of [L]ayers
L(L_N, 1) = Layer;

for l_i = 2:L_N
    L(l_i).b = readmatrix( strcat( 'b', num2str(l_i) ) );
    L(l_i).w = readmatrix( strcat( 'w', num2str(l_i) ) );
end

for train_round_i = 1:Def.TRAIN_ROUNDS_N
    for mini_batch_i = 1:(train_data_n / Def.MINI_BATCH_LENGTH)
        for l_i = 2:L_N
            L(l_i).dcdb_cum = zeros( size( L(l_i).b ) );
            L(l_i).dcdw_cum = zeros( size( L(l_i).w ) );
        end

        start_i = (mini_batch_i - 1) * Def.MINI_BATCH_LENGTH + 1;
        end_i = start_i + Def.MINI_BATCH_LENGTH - 1;
        for train_data_i = start_i:end_i
            r = to_r( train_ou(train_data_i) );

            % feedfoward
            L(1).y = transpose( train_in(train_data_i, :) );
            for l_i = 2:L_N
                L(l_i).z = L(l_i).w * L(l_i-1).y + L(l_i).b;
                L(l_i).y = af( L(l_i).z );
            end

            % backpropagation
            dcdb = ( 2 * ( L(L_N).y - r ) ) .* f_d( L(L_N).z );
            dcdw = dcdb * transpose( L(L_N - 1).y );
            L(L_N).dcdb_cum = L(L_N).dcdb_cum + dcdb;
            L(L_N).dcdw_cum = L(L_N).dcdw_cum + dcdw;
            for l_i = (L_N - 1):-1:2
                sub_dcdb = zeros( size( L(l_i).b ) ) + dcdb(1);
                sub_w    = zeros( size( L(l_i).b ) ) + L(l_i + 1).w(1, 1);
                dcdb = sub_dcdb .* sub_w .* f_d( L(l_i).z );
                dcdw = dcdb * transpose( L(l_i - 1).y );
                L(l_i).dcdb_cum = L(l_i).dcdb_cum + dcdb;
                L(l_i).dcdw_cum = L(l_i).dcdw_cum + dcdw;
            end
        end

        for l_i = 2:L_N
            L(l_i).b = L(l_i).b - L(l_i).dcdb_cum ./ Def.MINI_BATCH_LENGTH * Def.LEARNING_RATE;
            L(l_i).w = L(l_i).w - L(l_i).dcdw_cum ./ Def.MINI_BATCH_LENGTH * Def.LEARNING_RATE;
        end
    end
end

test_data = load('mnist_test.csv');
test_data_n = numel( test_data(:, 1) );
test_in = test_data(:, 2:785);
test_ou = test_data(:, 1);

correct_n = 0;
for test_data_i = 1:test_data_n
	L(1).y = transpose( test_in(test_data_i, :) );
    for l_i = 2:L_N
        L(l_i).z = L(l_i).w * L(l_i-1).y + L(l_i).b;
        L(l_i).y = af( L(l_i).z );
    end
    
    max = 0;
    out_predict = -1;
    for i = 1:10
        if L(L_N).y(i) > max
            max = L(L_N).y(i);
            out_predict = i - 1;
        end
    end
    
    if out_predict == test_ou(test_data_i)
        correct_n = correct_n + 1;
    end
end

accuracy = correct_n / test_data_n
