test_data = load('mnist_test.csv');
test_data_n = numel( test_data(:, 1) );
test_in = normalized_grayscale( test_data(:, 2:785) );
test_ou = test_data(:, 1);
load('trained_parameters', 'L');

correct_n = 0;
for test_data_i = 1:test_data_n
	L(1).y = transpose( test_in(test_data_i, :) );
    for l_i = 2:L_N
        L(l_i).z = L(l_i).w * L(l_i-1).y + L(l_i).b;
        L(l_i).y = f( L(l_i).z );
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