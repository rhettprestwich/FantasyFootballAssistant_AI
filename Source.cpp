#include "netCreator.h"

void theTest();
void anotherTest();

int main()
{

	//anotherTest();
	//return 0;
	/*
	std::wofstream myFile;
	myFile.open("NFL1.txt");
	FootballLeague NFL1(8);
	NFL1.getTheStats();
	NFL1.printLeague(myFile);
	myFile.close();

	FootballLeague NFL2(NFL1);
	std::wofstream myFile2;
	myFile2.open("NFL2.txt");
	NFL2.printLeague(myFile2);
	myFile2.close();
	return 0;
	*/

	double theAv = 0;
	int countAv = 0;
	for (int j = 5; j < 14; ++j)
	{
		FootballLeague NFL(j);
		NFL.getTheStats();

		std::wcout << j << L" WEEKS:" << std::endl;
		try
		{
			double accuracyTotal = 0;
			int numVal = 25;
			// Create NN using my personal class for convienience:
			for (int i = 0; i < numVal; ++i)
			{
				netCreator NC(NFL, j);
				accuracyTotal += NC.accPer;
			}
			std::wcout << std::endl << L"Total Accuracy: " << (accuracyTotal / static_cast<double>(numVal)) << std::endl;
			theAv += (accuracyTotal / static_cast<double>(numVal));
		}
		catch (const std::exception& e)
		{
			std::cout << e.what() << std::endl;
			return EXIT_FAILURE;
		}
		std::wcout << std::endl << std::endl;
		++countAv;
	}
	std::wcout << L"Complete Average Over " << countAv << L" Groups: " << (theAv / static_cast<double>(countAv)) << std::endl;

	return 0;
}

/*--- FUNCTIONS: ---*/
/*------------------*/

void anotherTest()
{
	FANN::training_data tempDat;
	const unsigned int num_data = 2; //the number of training data
	const unsigned int num_output = 1; //number of outputs per training data
	const unsigned int num_input = 2; //the number of inputs per training data

	double **tInput = new double*[2];
	double **output = new double*[2];
	for (int i = 0; i < 2; ++i)
	{
		tInput[i] = new double[2];
		output[i] = new double[1];
	}

	tInput[0][0] = 230.0 / 80000.0;
	tInput[0][1] = 0.0;
	tInput[1][0] = 79607.0 / 80000.0;
	tInput[1][1] = 13.0;

	double * heyThere = tInput[0];
	double helloAgain = heyThere[0];
	helloAgain = tInput[1][0];

	output[0][0] = 0.0;
	output[1][0] = 1.0;

	//create training data:
	FANN::training_data tData;
	tData.set_train_data(num_data, num_input, tInput, num_output, output);

	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 1;

	//create neural network:
	FANN::neural_net net;
	net.create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	FANN::training_algorithm_enum a = net.get_training_algorithm();

	//Change parameters of NN: 
	net.set_activation_function_output(FANN::LINEAR);
	net.set_training_algorithm(FANN::TRAIN_INCREMENTAL);
	net.print_parameters();

	tData.save_train("test tester.txt");


	const float desired_error = (const float) 0.000001; //the desired get_MSE or get_bit_fail, depending of which stop function is chosen by set_train_stop_function.
	const unsigned int max_epochs = 5000000; //The maximum number of epochs the training should continue
	const unsigned int epochs_between_reports = 10000;//The number of epochs between printing a status report to stdout.  A value of zero means no reports should be printed.

	net.set_learning_rate((const float)0.1);

	//train neural network:
	net.train_on_data(tData, max_epochs, epochs_between_reports, desired_error);

	std::wofstream myfile("PredictTEST.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < 2; ++i)
		{
			for (int j = 0; j < 2; ++j)
				myfile << tInput[i][j] << L" ";
			double * results = net.run(tInput[i]);
			myfile << L" --> " << *results << std::endl;
		}
	}

	tData.destroy_train();

	net.destroy();
}


void theTest()
{
	//TESTING:
	FANN::training_data tempDat;
	const unsigned int num_data = 4; //the number of training data
	const unsigned int num_output = 1; //number of outputs per training data
	const unsigned int num_input = 2; //the number of inputs per training data

	double **tInput = new double*[4];
	double **output = new double*[4];
	for (int i = 0; i < 4; ++i)
	{
		tInput[i] = new double[2];
		output[i] = new double[1];
	}
	tInput[0][0] = 0;
	tInput[0][1] = 0;
	tInput[1][0] = 4;
	tInput[1][1] = 0;
	tInput[2][0] = 0;
	tInput[2][1] = 4;
	tInput[3][0] = 4;
	tInput[3][1] = 4;

	double * heyThere = tInput[0];
	double helloAgain = heyThere[0];
	helloAgain = tInput[1][0];

	output[0][0] = 0;
	output[1][0] = 0;
	output[2][0] = 0;
	output[3][0] = 15;

	//create training data:
	FANN::training_data tData;
	tData.set_train_data(num_data, num_input, tInput, num_output, output);

	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 1;

	//create neural network:
	FANN::neural_net net;
	net.create_standard(num_layers, num_input, num_neurons_hidden, num_output);

	FANN::training_algorithm_enum a = net.get_training_algorithm();

	//Change parameters of NN: 
	net.set_activation_function_output(FANN::LINEAR);
	net.set_training_algorithm(FANN::TRAIN_INCREMENTAL);
	net.print_parameters();

	//tData.save_train("test tester.txt");


	const float desired_error = (const float) 0.00001; //the desired get_MSE or get_bit_fail, depending of which stop function is chosen by set_train_stop_function.
	const unsigned int max_epochs = 50000000; //The maximum number of epochs the training should continue
	const unsigned int epochs_between_reports = 10000;//The number of epochs between printing a status report to stdout.  A value of zero means no reports should be printed.

	net.set_learning_rate((const float)0.00001);

	//train neural network:
	net.train_on_data(tData, max_epochs, epochs_between_reports, desired_error);

	std::wofstream myfile("PredictTEST.txt");
	if (myfile.is_open())
	{
		for (int i = 0; i < 4; ++i)
		{
			for (int j = 0; j < 2; ++j)
				myfile << tInput[i][j] << L" ";
			double * results = net.run(tInput[i]);
			myfile << L" --> " << *results << std::endl;
		}
	}

	tData.destroy_train();

	net.destroy();



	//END TESTING
}