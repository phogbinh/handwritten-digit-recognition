#include <bits/stdc++.h>

using namespace std;

void he_gen(string file_name, int row_n, int col_n, double std_dev)
{
    ofstream file(file_name);
    random_device rd{};
    mt19937 gen{ rd() };
    normal_distribution<> nd{ 0.0, std_dev };
    for (int i = 0; i < row_n; ++i)
    {
        for (int j = 0; j < col_n; ++j)
        {
            if (j != 0) file << ",";
            file << nd(gen);
        }
        file << endl;
    }
    file.close();
}

int main()
{
    freopen("architect", "r", stdin);
    vector<int> l_n;
    int v;
    while (scanf("%d", &v) != EOF) l_n.push_back(v);
    for (int i = 1; i < (int)l_n.size(); ++i)
    {
        he_gen( "b" + to_string(i + 1), l_n[i], 1,        0.0                  );
        he_gen( "w" + to_string(i + 1), l_n[i], l_n[i-1], sqrt(2.0 / l_n[i-1]) );
    }
    return 0;
}
