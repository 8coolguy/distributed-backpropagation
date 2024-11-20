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
	SE cost;
	cout << f.evaluate(5.0) << endl;
	Layer a(1,1,1.0,&f,1.0);
	double b[1] = { 2 };
	a.forward(b);
	a.backward(b, &cost, .1);
	a.forward(b);
	a.backward(b, &cost, .1);
	a.forward(b);
	a.backward(b, &cost, .1);
	a.forward(b);
	a.backward(b, &cost, .1);
	a.forward(b);
	a.backward(b, &cost, .1);


}

