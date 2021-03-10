//Mark Boady - Drexel University - Homework Assignment

//Import iostream
#include <iostream>
//use namespace so we can type cout instead of std::cout
using namespace std;


//Compute and return x^y
//Use a while loop and repeated multiplication.
//This algorithm is from HW2
//For example exponent(3,4)=3^4=81
int exponent(int x, int y)
{
	if (y == 0)
		return 1;
	else
		return x * exponent(x, y-1);
}

//You don't need to make any changes to the code below here
int main()
{
	int base;
    int power;
    int result;
	cout << "Hello. Enter Values to compute exponent." << endl;
	cout << "Enter Base: ";
	cin >> base;
	cout << "Enter Exponent: ";
	cin >> power;
	result = exponent(base,power);
	cout << "The result is " << result << endl;
	return 0;
}
