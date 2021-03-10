//Mark Boady - Drexel University - Homework Assignment

//Import iostream
#include <iostream>
#include <string> //For working with text
//use namespace so we can type cout instead of std::cout
using namespace std;

//Print String in reverse
//Use a while loop to print a string in reverse order.
//Print a \n after the string so it looks pretty.
//Use C++ strings https://www.cplusplus.com/reference/string/string/

//This function should print the result using cout.
//Void means no return value given. (Since you print the answer)
//This is based on HW2
//For example reversePrint("Apple") would print elppA
//You will only ever get 1 word to reverse, not a multi-word sentence.
void reversePrint(string text)
{
	std::string::const_iterator p = text.cend();
	while (p != text.cbegin()){
		cout << *p;
		--p;
	}
	cout << *p << endl;
}
int main()
{
	string str;//Give a max size
	cout << "Hello. This code reverses a string." << endl;
	cout << "Text to reverse: ";
	cin >> str;
	reversePrint(str);
	return 0;
}
