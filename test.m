% input
TEST_DATA = load('mnist_test.csv');
TEST_DATA_N = numel( TEST_DATA(:, 1) );
TEST_IN = normalized_grayscale( TEST_DATA(:, 2:785) );
TEST_OU = TEST_DATA(:, 1);
clear TEST_DATA;
load('trained_parameters', 'layers');
LAYERS_N = numel(layers);
layers_associates(LAYERS_N, 1) = layer_associates;
correct_n = 0;

% test
for test_data_i = 1:TEST_DATA_N
    % feedforward
	layers_associates(1).y = transpose( TEST_IN(test_data_i, :) );
    for l_i = 2:LAYERS_N
        layers_associates(l_i).z = layers(l_i).w * layers_associates(l_i-1).y + layers(l_i).b;
        layers_associates(l_i).y = f( layers_associates(l_i).z );
    end
    
    max = 0;
    pred_ou = -1; % [pred]icted [ou]tput
    for i = 1:10
        if layers_associates(LAYERS_N).y(i) > max
            max = layers_associates(LAYERS_N).y(i);
            pred_ou = i - 1;
        end
    end
    
    if pred_ou == TEST_OU(test_data_i)
        correct_n = correct_n + 1;
    end
end

% output
disp( ['accuracy: ' num2str(correct_n / TEST_DATA_N * 100), '%'] );
clear;