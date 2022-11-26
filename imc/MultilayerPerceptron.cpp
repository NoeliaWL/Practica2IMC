/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	eta = 0.7;
	mu = 1.0;
	online = false;
	outputFunction = 0;
}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	nOfLayers = nl;
	layers = new Layer[nOfLayers];
	
	//cout << "-->" << nOfLayers << endl;

	for(int i=0; i<nOfLayers; i++){
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[layers[i].nOfNeurons];
	    //cout << "--->" << nOfLayers << " " << i << endl;

		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(i == 0 or (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1 and outputFunction == 1)){
				layers[i].neurons[j].w = NULL;
				layers[i].neurons[j].deltaW = NULL;
				layers[i].neurons[j].lastDeltaW = NULL;
				layers[i].neurons[j].wCopy = NULL;
			}
			else{
				layers[i].neurons[j].w = new double[npl[i-1] + 1];
				layers[i].neurons[j].deltaW = new double[npl[i-1] + 1];
				layers[i].neurons[j].lastDeltaW = new double[npl[i-1] + 1];
				layers[i].neurons[j].wCopy = new double[npl[i-1] + 1];
			}
		}
	}
	return 1;
}



// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for(int i=0; i<nOfLayers; i++){
		delete[] layers[i].neurons;
	}

	delete[] layers;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for(int i=1; i<nOfLayers; i++){
		for (int j=0; j<layers[i].nOfNeurons; j++){
			for(int k=0; k<layers[i-1].nOfNeurons+1; k++){
				if(outputFunction == 1){
					if(not (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
						layers[i].neurons[j].w[k] = randomDouble(-1, 1);
						layers[i].neurons[j].deltaW[k] = 0.0;
						layers[i].neurons[j].lastDeltaW[k] = 0.0;
						layers[i].neurons[j].wCopy[k] = 0.0;
					}
				}
				else if(outputFunction == 0){
					layers[i].neurons[j].w[k] = randomDouble(-1, 1);
					layers[i].neurons[j].deltaW[k] = 0.0;
					layers[i].neurons[j].lastDeltaW[k] = 0.0;
					layers[i].neurons[j].wCopy[k] = 0.0;
				}
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for(int i=0; i<layers[0].nOfNeurons; i++){
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output) {
	for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++){
		output[i] = layers[nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for(int i=1; i<nOfLayers; i++){
		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(not (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
				for(int k=0; k<layers[i-1].nOfNeurons + 1; k++){
						layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
				}
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for(int i=1; i<nOfLayers; i++){
		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(not (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
				for(int k=0; k<layers[i-1].nOfNeurons + 1; k++){
						layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
				}
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	double net;

	if(outputFunction == 0){
		//SALIDAS - ULTIMA CAPA SIGMOIDE
		for(int i=0; i<nOfLayers; i++){
			for(int j=0; j<layers[i].nOfNeurons; j++){
				net = 0.0;
				for(int k=1; k<layers[i-1].nOfNeurons+1; k++){
					net += layers[i].neurons[j].w[k] * layers[i-1].neurons[k-1].out;
				}
				net += layers[i].neurons[j].w[0];
				layers[i].neurons[j].out = 1.0/(1 + exp(-net));
			}
		}
	}
	else if(outputFunction == 1){
		//SALIDAS - ULTIMA CAPA SOFTMAX
		double netLast;

		for(int i=1; i<nOfLayers; i++){
			netLast = 0.0;
			for(int j=0; j<layers[i].nOfNeurons; j++){
				net = 0.0;
			    if(not (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
					for(int k=1; k<layers[i-1].nOfNeurons+1; k++){
						net += layers[i].neurons[j].w[k] * layers[i-1].neurons[k-1].out;
					}
					net += layers[i].neurons[j].w[0];
				}
				if(i == nOfLayers-1){
					layers[i].neurons[j].out = exp(net);
					netLast += exp(net);
				}
				else{
					layers[i].neurons[j].out = 1.0/(1 + exp(-net));
				}
			}

			if(i == nOfLayers-1){
				for(int j=0; j<layers[i].nOfNeurons; j++){
					layers[i].neurons[j].out /= netLast;
				}
			}
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {
	double error = 0.0;

	if(errorFunction == 0){
		//MSE
		for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++){
			error += pow(layers[nOfLayers-1].neurons[i].out - target[i], 2);
		}

		error /= layers[nOfLayers-1].nOfNeurons;
	}
	else if(errorFunction == 1){
		//Cross Entropy
		for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++){
			error += target[i] * log(layers[nOfLayers-1].neurons[i].out);
		}

		error /= layers[nOfLayers-1].nOfNeurons;
	}

	return (double) error;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {
	if(outputFunction == 0){ //SIGMOIDE EN ULTIMA CAPA
		for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++){
			if(errorFunction == 0){ // ERROR MSE
				layers[nOfLayers-1].neurons[i].delta = -(target[i] - layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[i].out * (1 - layers[nOfLayers-1].neurons[i].out);
			}
			else{ // CROSS ENTROPY
				layers[nOfLayers-1].neurons[i].delta = -(target[i] / layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[i].out * (1 - layers[nOfLayers-1].neurons[i].out);
			}
		}
	}
	else if(outputFunction == 1){ //SOFTMAX EN ULTIMA CAPA
		double sumDeltaSalida = 0.0;
		int condicionSoftmax;
		for(int j=0; j<layers[nOfLayers-1].nOfNeurons; j++){
			sumDeltaSalida = 0.0;
			for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++){
				if(i == j){
					condicionSoftmax = 1;
				}
				else{
					condicionSoftmax = 0;
				}
				if(errorFunction == 0){ // ERROR MSE
					sumDeltaSalida += (target[i] - layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[j].out * (condicionSoftmax - layers[nOfLayers-1].neurons[i].out);
				}
				else{ // CROSS ENTROPY
					sumDeltaSalida += (target[i] / layers[nOfLayers-1].neurons[i].out) * layers[nOfLayers-1].neurons[j].out * (condicionSoftmax - layers[nOfLayers-1].neurons[i].out);
				}
			}
			layers[nOfLayers-1].neurons[j].delta = -sumDeltaSalida;
		}
	}

	double sumDelta;

	for(int h=nOfLayers-2; h>0; h--){
		for(int j=0; j<layers[h].nOfNeurons; j++){
			sumDelta = 0.0;
			for(int i=0; i<layers[h+1].nOfNeurons; i++){
			    if(not (h+1 == nOfLayers-1 and i == layers[nOfLayers-1].nOfNeurons-1)){
					sumDelta += layers[h+1].neurons[i].w[j+1] * layers[h+1].neurons[i].delta;
				}
			}
			layers[h].neurons[j].delta = sumDelta * layers[h].neurons[j].out * (1 - layers[h].neurons[j].out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for(int h=1; h<nOfLayers; h++){
		for(int j=0; j<layers[h].nOfNeurons; j++){
			if(not (h == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
				for(int i=1; i<layers[h-1].nOfNeurons+1; i++){
					layers[h].neurons[j].deltaW[i] += layers[h].neurons[j].delta * layers[h-1].neurons[i-1].out;
				}
				layers[h].neurons[j].deltaW[0] += layers[h].neurons[j].delta;
			}
		}
	}
}

// ------------------------------
// Restore to 0 the deltaW
void MultilayerPerceptron::accumulateChangeRestore(){
	for(int h=1; h<nOfLayers; h++){
		for(int j=0; j<layers[h].nOfNeurons; j++){
			if(not (h == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
				for(int i=1; i<layers[h-1].nOfNeurons+1; i++){
					layers[h].neurons[j].lastDeltaW[i] = layers[h].neurons[j].deltaW[i];
					layers[h].neurons[j].deltaW[i] = 0;
				}
				layers[h].neurons[j].lastDeltaW[0] = layers[h].neurons[j].deltaW[0];
				layers[h].neurons[j].deltaW[0] = 0;
			}
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	for(int h=1; h<nOfLayers; h++){
		for(int j=0; j<layers[h].nOfNeurons; j++){
			if(not (h == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
			    for(int i=1; i<layers[h-1].nOfNeurons+1; i++){
				    if(online){
					    layers[h].neurons[j].w[i] = layers[h].neurons[j].w[i] - (eta * layers[h].neurons[j].deltaW[i]) - (mu * (eta * layers[h].neurons[j].lastDeltaW[i]));
				    }
				    else{
					    layers[h].neurons[j].w[i] = layers[h].neurons[j].w[i] - (eta * layers[h].neurons[j].deltaW[i])/nOfTrainingPatterns - (mu * (eta * layers[h].neurons[j].lastDeltaW[i]))/nOfTrainingPatterns;
				    }
			    }
			    if(online){
				    layers[h].neurons[j].w[0] = layers[h].neurons[j].w[0] - (eta * layers[h].neurons[j].deltaW[0]) - (mu * (eta * layers[h].neurons[j].lastDeltaW[0]));
			    }
			    else{
				    layers[h].neurons[j].w[0] = layers[h].neurons[j].w[0] - (eta * layers[h].neurons[j].deltaW[0])/nOfTrainingPatterns - (mu * (eta * layers[h].neurons[j].lastDeltaW[0]))/nOfTrainingPatterns;
			    }
			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for(int i=1; i<nOfLayers; i++){
		cout << "Layer " << i <<endl;
		for(int j=0; j<layers[i].nOfNeurons; j++){
			if(not (i == nOfLayers-1 and j == layers[nOfLayers-1].nOfNeurons-1)){
				for(int k=0; k<layers[i-1].nOfNeurons+1; k++){
					cout << layers[i].neurons[j].w[k] << " ";
				}
				cout << endl;
			}
		}
		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {
	if(online){
		accumulateChangeRestore();
	}
	
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	accumulateChange();

	if(online){
		weightAdjustment();
	}
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	if(!online){
		accumulateChangeRestore();
	}
	for(int i=0; i<trainDataset->nOfPatterns; i++){
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
	}
	if(!online){
		weightAdjustment();
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	double error = 0.0;

	for(int i=0; i<dataset->nOfPatterns; i++){
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		error += obtainError(dataset->outputs[i], errorFunction);
	}

	error /= dataset->nOfPatterns;
	if(errorFunction == 1){
		error *= -1;
	}

	return error;
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
	int CCR = 0.0;
	double *salidas = new double[layers[nOfLayers-1].nOfNeurons];

	for(int i=0; i<dataset->nOfPatterns; i++){
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

	    int maxIndexSalidaReal = 0, maxIndexSalidaObtenida = 0;
		for(int j=1; j<dataset->nOfOutputs; j++){
			if(dataset->outputs[i][j] == 1){
				maxIndexSalidaReal = j;
			}

			if(salidas[j] > salidas[maxIndexSalidaObtenida]){
				maxIndexSalidaObtenida = j;
			}
		}

		if(maxIndexSalidaReal == maxIndexSalidaObtenida){
			CCR++;
		}
	}

	CCR = (double) 100 * (CCR / dataset->nOfPatterns);

	return CCR;
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;


	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	if ( iterWithoutImproving!=50)
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (k==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
