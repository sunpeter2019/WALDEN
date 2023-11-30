#include <walden.h>
#include <iostream>
#include <vector>
#include <cassert>

using namespace std;

/*
    this shows a simple example of walden's bulkload operation
*/

int main()
{
    WALDEN<int, int, int> walden;

    // prepare data
    vector<std::tuple<int, int,int >> data;
    for (int i = 0; i < 100000; i ++) {
        data.push_back({i, i % 127,1});
    }

    // bulk load
    walden.bulk_load(data.data(), data.size());

    // normal insert
    walden.insert(-100, 5,1);
    walden.insert(187688, 4 ,1);

    // verify walden data structure
    walden.verify();

    // test correctness
    assert(walden.at(-100) == 5);
    assert(walden.at(187688) == 4);
    for (int i = 0; i < 100000; i ++) {
        assert(walden.exists(i));
        assert(walden.at(i) == (i % 127));
    }

    cout << "bolk load success" << endl;

    return 0;
}
