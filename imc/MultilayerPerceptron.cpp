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

int seed = 0;

int** confusionMatrix;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	nOfLayers = 1;
	mu = 1;
	eta = 0.7;
	online = false;
	outputFunction = 0;
}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	nOfLayers = nl;
	layers = new Layer[nl];
	
	for(int i=0; i<nOfLayers; i++) {
		layers[i].nOfNeurons = npl[i];
		layers[i].neurons = new Neuron[npl[i]];
	}
	
	for(int i=0; i<nOfLayers; i++) {
		for(int j=0; j<npl[i]; j++) {
			if(i == 0) {
				layers[i].neurons[j].w = NULL;
				layers[i].neurons[j].deltaW = NULL;
				layers[i].neurons[j].wCopy = NULL;
				layers[i].neurons[j].lastDeltaW = NULL;
			}
			else {
				layers[i].neurons[j].w = new double[npl[i-1] + 1];
				layers[i].neurons[j].deltaW = new double[npl[i-1] + 1];
				layers[i].neurons[j].wCopy = new double[npl[i-1] + 1];
				layers[i].neurons[j].lastDeltaW = new double[npl[i-1] + 1];
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
	for(int i=0; i<nOfLayers; i++) {
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			delete[] layers[i].neurons[j].w;
			delete[] layers[i].neurons[j].deltaW;
			delete[] layers[i].neurons[j].wCopy;
		}
		delete[] layers[i].neurons;
	}
	delete[] layers;
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for(int i=0; i<nOfLayers; i++) {
		for(int j=0; j<layers[i].nOfNeurons;j++) {
			for(int k=0; k<layers[i-1].nOfNeurons; k++) {
				layers[i].neurons[j].w[k] = randomDouble(-1, 1);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for(int i=0; i<layers[0].nOfNeurons; i++) {
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output) {
	for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++) {
		output[i] = layers[nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for(int i=0; i<nOfLayers; i++) {
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			for(int k=0; k<layers[i-1].nOfNeurons; k++) {
				layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for(int i=1; i<nOfLayers; i++) {
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			for(int k=0; k<layers[i-1].nOfNeurons + 1; k++) {
				layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	double net;
	double sumNet = 0.0;
	
	for(int i=1; i<nOfLayers; i++) {
		sumNet = 0.0;
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			net = 0.0;
			for(int k=1; k<layers[i-1].nOfNeurons + 1; k++) {
				net += layers[i].neurons[j].w[k] * layers[i-1].neurons[k-1].out;
			}
			
			net += layers[i].neurons[j].w[0];
			
			if(i == (nOfLayers - 1) && outputFunction == 1) {
				layers[i].neurons[j].out = exp(net);
				sumNet += exp(net);
			}
			else {
				layers[i].neurons[j].out = 1.0 / (1 + exp(-net));
			}
		}
		
		if(i == (nOfLayers - 1) && outputFunction == 1) {
			for(int j=0; j<layers[i].nOfNeurons; j++) {
				layers[i].neurons[j].out /= sumNet;
			}
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {
	if(errorFunction == 0) {
		double mse = 0.0;
		for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++) {
			mse += pow(target[i] - layers[nOfLayers - 1].neurons[i].out, 2);
		}
		
		return mse /= (double) layers[nOfLayers-1].nOfNeurons;
	}
	
	double entropy = 0.0;
	for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++) {
		entropy += target[i] * log(layers[nOfLayers-1].neurons[i].out);
	}
	
	return entropy / (double) layers[nOfLayers-1].nOfNeurons;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {
	double out, aux;
	
	for(int i=0; i<layers[nOfLayers-1].nOfNeurons; i++) {
		out = layers[nOfLayers-1].neurons[i].out;
		layers[nOfLayers-1].neurons[i].delta = 0.0;
		
		if(outputFunction == 1) { //Funcion Softmax
			int conditionSoftmax = 0;
			
			for(int j=0; j<layers[nOfLayers-1].nOfNeurons; j++) {
				if(j == i) {
					conditionSoftmax = 1;
				}
				else{
					conditionSoftmax = 0;
				}
				
				if(errorFunction == 0) {
					layers[nOfLayers-1].neurons[i].delta += -(target[j] - layers[nOfLayers-1].neurons[j].out) * out * (conditionSoftmax - layers[nOfLayers-1].neurons[j].out);
				}
				else {
					layers[nOfLayers-1].neurons[i].delta += -(target[j] / layers[nOfLayers - 1].neurons[j].out) * out * (conditionSoftmax - layers[nOfLayers - 1].neurons[j].out);
				}
			}
		}
		else { //Funcion Sigmoide
			if(errorFunction == 0) {
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] - out) * out * (1 - out);
			}
			else {
				layers[nOfLayers - 1].neurons[i].delta = -(target[i] / out) * out * (1 - out);
			}
		}
	}
	
	for(int i=nOfLayers-2; i>=1; i--) {
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			out = layers[i].neurons[j].out;
			aux = 0.0;
			for(int k=0; k<layers[i+1].nOfNeurons; k++) {
				aux += layers[i+1].neurons[k].w[j+1] * layers[i+1].neurons[k].delta;
			}
			
			layers[i].neurons[j].delta = aux * out * (1 - out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for(int i=1; i<nOfLayers; i++) {
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			for(int k=1; k<layers[i-1].nOfNeurons + 1; k++) {
				layers[i].neurons[j].deltaW[k] += layers[i].neurons[j].delta * layers[i-1].neurons[k-1].out;
			}
			
			layers[i].neurons[j].deltaW[0] += layers[i].neurons[j].delta;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	if(online) { //Version online
		for(int i=1; i<nOfLayers; i++) {
			for(int j=1; j<layers[i].nOfNeurons; j++) {
				for(int k=1; k<layers[i-1].nOfNeurons + 1; k++) {
					layers[i].neurons[j].w[k] += (eta * layers[i].neurons[j].deltaW[k]) - (mu * eta * layers[i].neurons[j].lastDeltaW[k]);
				}
				
				layers[i].neurons[j].w[0] += (eta * layers[i].neurons[j].deltaW[0]) - (mu * eta * layers[i].neurons[j].lastDeltaW[0]);
			}
		}
	}
	else { //Version offline
		for(int i=1; i<nOfLayers; i++) {
			for(int j=1; j<layers[i].nOfNeurons; j++) {
				for(int k=1; k<layers[i-1].nOfNeurons; k++) {
					layers[i].neurons[j].w[k] -= (eta * layers[i].neurons[j].deltaW[k] / nOfTrainingPatterns) - (mu * (eta * layers[i].neurons[j].lastDeltaW[k]) / nOfTrainingPatterns);
				}
				
				layers[i].neurons[j].w[0] -= (eta * layers[i].neurons[j].deltaW[0] / nOfTrainingPatterns) - (mu * (eta * layers[i].neurons[j].lastDeltaW[0]) / nOfTrainingPatterns);
			}
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for(int i=1; i<nOfLayers; i++) {
		cout << "Layer " << i << endl;
		for(int j=0; j<layers[i].nOfNeurons; j++) {
			for(int k=0; k<layers[i].nOfNeurons + 1; k++) {
				cout << layers[i].neurons[j].w[k] << " ";
			}
			cout << endl;
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
	if(online) {
		for(int i=1; i<nOfLayers; i++) {
			for(int j=0; j<layers[i].nOfNeurons; j++) {
				for(int k=0; k<layers[i-1].nOfNeurons + 1; k++) {
					layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}
	
	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	accumulateChange();
	
	if(online) {
		weightAdjustment();
	}
}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	if(!online) {
		for(int i=1;i<nOfLayers; i++) {
			for(int j=0; j<layers[i].nOfNeurons; j++) {
				for(int k=0; k<layers[i-1].nOfNeurons + 1; k++) {
					layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}
	
	for(int i=0; i<trainDataset->nOfPatterns; i++) {
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
	}
	
	if(!online) {
		weightAdjustment();
	}
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	double sum = 0.0;
	
	for(int i=0; i<dataset->nOfPatterns; i++) {
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		
		sum += obtainError(dataset->outputs[i], errorFunction);
	}
	
	if(errorFunction == 0) {
		return sum / dataset->nOfPatterns; //MSE
	}
	
	return -1 * (sum / dataset->nOfPatterns); //Cross Entropy
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
	int ccr = 0.0;
	int expectedClass = 0, obtainedClass = 0;
	double *outArray = new double[layers[nOfLayers-1].nOfNeurons];
	double maximo = 0.0, maximo2 = 0.0;
	
	for(int i=0; i<dataset->nOfPatterns; i++) {
		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(outArray);
		
		maximo = outArray[0];
		maximo2 = dataset->outputs[i][0];
		for(int j=1; j<dataset->nOfOutputs; j++) {
			if(maximo < outArray[j]) {
				maximo = outArray[j];
				obtainedClass = j;
			}
			
			if(maximo2 < dataset->outputs[i][j]) {
				maximo2 = dataset->outputs[i][j];
				expectedClass = j;
			}
		}
		
		if(expectedClass == obtainedClass) {
			ccr++;
		}
		
		confusionMatrix[expectedClass][obtainedClass]++;
	}
	
	delete[] outArray;
	
	return ((double) ccr / dataset->nOfPatterns) * 100; 
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
	double trainCCR = 0.0;
	double testCCR = 0.0;
	
	confusionMatrix = new int*[5];
	for(int i=0; i<trainDataset->nOfOutputs; i++) {
		confusionMatrix[i] = new int[trainDataset->nOfOutputs];
	}
	
	string nameProblem;
	ostringstream aux;
	aux << "seed_" << seed << ".txt";
	nameProblem = aux.str();
	seed++;
	ofstream fichero(nameProblem);
	fichero << "Epoch\tTrainError\tETrainCCR\tTestCCR" << endl;
	
	cout<<""<<endl;
	
	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		trainCCR = testClassification(trainDataset);
		testCCR = testClassification(testDataset);
		
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
		fichero << countTrain << " " << trainError << " " << trainCCR << " " << testCCR << endl;

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
	
	fichero << endl << "Confusion Matrix[ExpectedClass][ObtainedClass]" << endl;
	for(int i=0; i<trainDataset->nOfOutputs; i++) {
		for(int j=0; j<trainDataset->nOfOutputs; j++) {
			fichero << confusionMatrix[i][j] << "\t";
		}
		fichero << endl;
	}
	
	delete confusionMatrix;
	
	fichero.close();
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
