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
    freopen("architect", "r", stdin);
    srand(1);
    vector<int> l_n;
    int u;
    while (scanf("%d", &u) != EOF) l_n.push_back(u);
    for (int i = 2; i <= (int)l_n.size(); ++i)
    {
        gen("b" + to_string(i), l_n[i-1], 1);
        gen("w" + to_string(i), l_n[i-1], l_n[i-2]);
    }
    return 0;
}
