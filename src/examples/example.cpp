#include <walden.h>
#include <iostream>

using namespace std;


/*
    this shows a simple example of walden
*/
int main()
{
    WALDEN<int, int, double> walden;

    // insert key-values-weight
    walden.insert(1, 1, 1);
    walden.insert(5, 5, 1);
    walden.insert(10, 10, 1 );

    cout << "exists(1) = " << (walden.exists(1) ? "true" : "false") << endl;
    cout << "exists(4) = " << (walden.exists(4) ? "true" : "false") << endl;

    cout << walden.at(5) << endl;

    // show tree structure
    walden.show();

    return 0;
}
