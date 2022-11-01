//============================================================================
// Introduction to computational models
// Name        : la2.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;

int main(int argc, char **argv) {
	// Process the command line
    bool Tflag = 0, wflag = 0, pflag = 0, tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0, oflag = 0, fflag = 0, sflag = 0, nflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue = NULL, *ivalue = NULL, *lvalue = NULL, *hvalue = NULL, *evalue = NULL, *mvalue = NULL, *fvalue = NULL, *svalue = NULL;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "T:w:t:i:l:h:e:m:f:s:o::n::p")) != -1)
    {

        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = optarg;
                break;
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
            case 'o':
                oflag = true;
                break;
            case 'f':
                fflag = true;
                fvalue = optarg;
                break;
            case 's':
                sflag = true;
                svalue = optarg;
                break;
            case 'n':
                nflag = true;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt == 't' || optopt == 'i' || optopt == 'l' || optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'f' || optopt == 's')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }


    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp. For example, mlp.eta = value
        mlp.online = oflag;
        if(eflag){
            mlp.eta = atof(evalue);
        }

        if(mflag){
            mlp.mu = atof(mvalue);
        }

        if(sflag){
            mlp.outputFunction = atof(svalue);
        }

    	// Type of error considered
    	int error = 0;
        if(fflag){
            error = atof(fvalue);
        }

    	// Maximum number of iterations
    	int maxIter = 1000;
        if(iflag){
            maxIter = atof(ivalue);
        }

        // Read training and test data: call to util::readData(...)
    	Dataset * trainDataset = readData(tvalue);
    	Dataset * testDataset = NULL;
        if(Tflag){
            testDataset = readData(Tvalue);
        }
        else{
            testDataset = readData(tvalue);
        }

        // Initialize topology vector
        int layers = 1;
        if(lflag and atoi(lvalue) > 0){
            layers = atoi(lvalue);
        }

        int neurons = 4;
        if(hflag and atoi(hvalue) > 0){
            neurons = atoi(hvalue);
        }

        int *topology = new int[layers + 2];
        topology[0] = trainDataset->nOfInputs;
        for(int i=1; i<(layers+2-1); i++)
            topology[i] = neurons;
        topology[layers+2-1] = trainDataset->nOfOutputs;
        
        mlp.initialize(layers+2,topology);

		// Seed for random numbers
		int seeds[] = {1,2,3,4,5};
		double *trainErrors = new double[5];
		double *testErrors = new double[5];
		double *trainCCRs = new double[5];
		double *testCCRs = new double[5];
		double bestTestError = DBL_MAX;
		for(int i=0; i<5; i++){
			cout << "**********" << endl;
			cout << "SEED " << seeds[i] << endl;
			cout << "**********" << endl;
			srand(seeds[i]);
			mlp.runBackPropagation(trainDataset,testDataset,maxIter,&(trainErrors[i]),&(testErrors[i]),&(trainCCRs[i]),&(testCCRs[i]),error);
			cout << "We end!! => Final test CCR: " << testCCRs[i] << endl;

			// We save the weights every time we find a better model
			if(wflag && testErrors[i] <= bestTestError)
			{
				mlp.saveWeights(wvalue);
				bestTestError = testErrors[i];
			}
		}


		double trainAverageError = 0, trainStdError = 0;
		double testAverageError = 0, testStdError = 0;
		double trainAverageCCR = 0, trainStdCCR = 0;
		double testAverageCCR = 0, testStdCCR = 0;

        // Obtain training and test averages and standard deviations

		cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

		cout << "FINAL REPORT" << endl;
		cout << "*************" << endl;
	    cout << "Train error (Mean +- SD): " << trainAverageError << " +- " << trainStdError << endl;
	    cout << "Test error (Mean +- SD): " << testAverageError << " +- " << testStdError << endl;
	    cout << "Train CCR (Mean +- SD): " << trainAverageCCR << " +- " << trainStdCCR << endl;
	    cout << "Test CCR (Mean +- SD): " << testAverageCCR << " +- " << testStdCCR << endl;
		return EXIT_SUCCESS;
    } else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////

        // You do not have to modify anything from here.
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;

	}
}

