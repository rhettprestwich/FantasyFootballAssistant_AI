#ifndef NETCREATOR_H
#define NETCREATOR_H

#include "ApiData.h"
#include <doublefann.h>
#include <fann_cpp.h>
#include <deque>

class netCreator
{
private:
	bool testing;		//--True when wanting to pull from stored stats vs. api stats
	bool alterations;	//--True when running a temporary alteration of program for analysis
	bool overRide;
	int week;			//--The lastest week that stats are available
	double validationPercent; //--The percent of the set that should be withheld durring training in order to test the network

	FootballLeague NFL; //the data from the NFL
	FANN::neural_net net; //the neural network structure

	double ** input; //the temporary input array, used to create the training data structure and the prediction input array
	double ** output; //the temporary output array, used to create the training data structure
	double ** testInput; //the data used in our validation
	double ** testOutput; //the data used in our validation
	double ** pInput; //the data to use to retrieve our predition

	unsigned int num_data; //the number of training data sets
	unsigned int num_output; //number of outputs per training data
	unsigned int num_input; //the number of inputs per training data
	unsigned int num_layers; //the total number of layers including the input and the output layer
	unsigned int num_neurons_hidden; //number of neurons in the hidden layer
	float desired_error; //the desired get_MSE or get_bit_fail, depending of which stop function is chosen by set_train_stop_function.
	unsigned int max_epochs; //The maximum number of epochs the training should continue
	unsigned int epochs_between_reports; //The number of epochs between printing a status report to stdout.  A value of zero means no reports should be printed.
	unsigned int num_data_test; //the number of data sets used for testing

public:
	netCreator(int inputWeek = 5); //the constructor goes through the process of building the Neural Network
	void startUp(bool NFLcopied);
	netCreator(FootballLeague NFLin, int in_week);

	void reviewAPIData(); //prints the NFL data structure that our data sets are built from
	double ** buildInput(); //extracts NFL data and processes it into sets of inputs. Returns both training and prediction data
	double ** buildOutput(); //extracts NFL data and processes it into sets of outputs.
	//double ** readPredFile();
	void weedOutBadCases(double **& ins, double **& outs); //refines the input and output arrays that we will use to create our training structure.
	void weedOutBadCases(FANN::training_data & T); //overloaded function, used when reading data from file, ie: testing == true
	void weedOutBadStats(double **& ins, double **& outs); //the inputs of every player, cutting out the least used ones in order to decrease our input nodes
	void getPrediction(); //uses the prediction inputs and the trained neural network to produce a prediction for each player.
	void createTrainingData(FANN::training_data & tData); //creates the training data structure
	void createNeuralNetwork(FANN::training_data & tData); //creates the neural network
	void setGenParams(); //explicitly set general parameters
	void setNetworkParams(); //explicitly set network params
	void setImpliedParams(); //implicitly set parameters, based on explicit ones
	std::vector<int> alterIds;
	std::vector<double *> setsUsed;
	void testSimple();
	void scaleStats(double **& ins, double **& outs);
	void splitSets(double **& ins, double **& outs);
	void validateNN(); //returns an accuracy percentage

	double accPer;
	double falsePositives;
	double falseNegatives;
};

#endif