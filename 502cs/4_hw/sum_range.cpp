//Mark Boady - Drexel University - Homework Assignment

//Import iostream
#include <iostream>
//use namespace so we can type cout instead of std::cout
using namespace std;

//Sum Numbers in a Range
//Use a for/while loop to sum all numbers in a range
//This is based on an algorithm from HW2
//Sum all numbers from x to y
//For example sumRange(2,5)=2+3+4+5=14
int sumRange(int x, int y)
{
	int sum = 0;
	for(int i = x; i <= y; i++)
		sum += i;
	return sum;
}
int main()
{
	int start;
    int stop;
    int result;
	cout << "Hello." << endl;
	cout << "Enter Starting Value: ";
	cin >> start;
	cout << "Enter Stopping Point: ";
	cin >> stop;
	result = sumRange(start,stop);
	cout << "The result is " << result << endl;
	return 0;
}
