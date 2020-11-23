#include <bits/stdc++.h>

using namespace std;

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void gen(string file_name, int row_n, int col_n)
{
    ofstream file(file_name);
    for (int i = 0; i < row_n; ++i)
    {
        for (int j = 0; j < col_n; ++j)
        {
            if (j != 0) file << ",";
            file << fRand(-0.05, 0.05);
        }
        file << "\n";
    }
    file.close();
}

int main()
{
    for (int l1 = 16; l1 < 100; ++l1)
        for (int l2 = 16; l2 < 100; ++l2)
        {
            string suffix = "_" + to_string(l1) + "_" + to_string(l2) + ".t";
            freopen(string("architect" + suffix).c_str(), "r", stdin);
            srand(1);
            vector<int> l_n;
            int u;
            while (scanf("%d", &u) != EOF) l_n.push_back(u);
            for (int i = 2; i <= (int)l_n.size(); ++i)
            {
                gen("b" + to_string(i) + suffix, l_n[i-1], 1);
                gen("w" + to_string(i) + suffix, l_n[i-1], l_n[i-2]);
            }
        }
    return 0;
}
