/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include <iostream>
#include "Activation_Function.h"

using namespace std;

int main(void){
	Sigmoid f;
	cout << f.evaluate(5.0) << endl;
}

