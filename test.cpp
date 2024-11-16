/*
 *
 * main.cpp
 * Main used for testing and measuring performance.
 *
 */ 

#include "Layer.h"
#include <iostream>
#include "Activation_Function.h"

using namespace std;

int main(void){
	Sigmoid f;
	cout << f.evaluate(5.0) << endl;
	Layer a(5,1,1.0,&f,1.0);
	double b[5] = { .1, .2, .3, .044, .05 };
	a.forward(b);
}

