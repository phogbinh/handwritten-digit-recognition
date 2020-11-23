#include <bits/stdc++.h>

using namespace std;

void gen_architect(int l1_neurons_n, int l2_neurons_n)
{
    ofstream file( "architect_" + to_string(l1_neurons_n) + "_" + to_string(l2_neurons_n) + ".t" );
    file << 784 << endl;
    file << l1_neurons_n << endl;
    file << l2_neurons_n << endl;
    file << 10 << endl;
    file.close();
}

int main()
{
    for (int i = 16; i < 100; ++i)
        for (int j = 16; j < 100; ++j)
            gen_architect(i, j);
    return 0;
}
