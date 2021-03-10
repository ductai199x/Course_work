//Mark Boady - Drexel University - Homework Assignment

//Import iostream
#include <iostream>
//use namespace so we can type cout instead of std::cout
using namespace std;

//Recursive Add Up
//This function MUST be defined recursively
//Implement the ADDUP function from HW3
/*
 Function addUp(n)
     If(n==0)
         return 0
     Else
         return n + addUp(n-1)
     EndIf
 EndFunction
 */
int addUp(int n)
{
    if (n == 0) 
	    return 0;
    else
        return n + addUp(n-1);
}

//You don't need to make any changes to the code below here
int main()
{
	int num;
    int result;
	
    cout << "Hello." << endl;
	cout << "This program recursively adds up numbers from 0 to n";
    cout << endl;
	cout << "Enter Number for n: ";
	cin >> num;
	result = addUp(num);
	cout << "Result is " << result << endl;
	return 0;
}
