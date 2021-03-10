//Mark Boady - Drexel University - Homework Assignment

//Import iostream
#include <iostream>
//use namespace so we can type cout instead of std::cout
using namespace std;


//Print in Binary
//Print the number given as a 8 bit binary number.
//Print all 8 bits.
//Assume the user will only give values from 0 to 255.
//You may use remainders, booleans, or shift commands.
//Print a newline after the binary value to make it easy to read.
//This based on a question from HW2
//Use cout to print the values in the function
//printBinary(75) should print 01001011
void printBinary(int numToConvert)
{
	int i = 7;
	char bits [8];
	while (i >= 0) {
		bits[i] = numToConvert % 2 + 48;
		numToConvert = numToConvert >> 1;
		i--;
	}

	cout << bits << endl;
}

int main()
{
	int num;
	cout << "Hello." << endl;
	cout << "This program only works with numbers from 0 to 255." << endl;
	cout << "Enter Number to Convert to Binary: ";
	cin >> num;
	printBinary(num);
	cout << endl;
	return 0;
}
