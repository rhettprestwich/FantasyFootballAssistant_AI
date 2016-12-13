#include "netCreator.h"

void theTest();
void anotherTest();
void analyse_data();

int main()
{
	int current_week = 14;
	std::wcout << L"Current Week in NFL? ";
	std::cin >> current_week;
	current_week -= 1;
	FootballLeague NFL(current_week);
	NFL.getTheStats();
	try
	{
		double accuracyTotal = 0.0;
		int numVal = 0;
		// Create NN using my personal class for convienience:
		for (int i = 0; i < numVal; ++i)
		{
			netCreator NC(NFL, current_week, i, 0.20);
			accuracyTotal += NC.accPer;
		}
		std::wcout << L"10-Fold Validations Yields " << (accuracyTotal / static_cast<double>(numVal)) << L"% Accuracy." << std::endl;
		std::wcout << std::endl << L"	~Generating Predictions..." << std::endl;
		netCreator predictionNet(NFL, current_week, 25, 0.00);
	}
	catch (const std::exception& e)
	{
		std::cout << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return 0;
}

/*--- FUNCTIONS: ---*/
/*------------------*/
void analyse_data()
{
	double theAv = 0;
	int countAv = 0;
	double theAvfP = 0.0, theAvfN = 0.0, theAvtP = 0.0, theAvtN = 0.0; //the average of all the different training sizes.
	int Av_num_pos = 0;
	int Av_num_neg = 0;
	int Av_num_tp = 0;
	int Av_num_fp = 0;
	int Av_num_tn = 0;
	int Av_num_fn = 0;

	std::wofstream resultFile;
	resultFile.open("result_range.csv");
	for (int j = 5; j < 14; ++j)
	{
		FootballLeague NFL(j);
		NFL.getTheStats();

		std::wcout << j << L" WEEKS:" << std::endl;
		resultFile << j << L" WEEKS:" << std::endl;

		double lowest_ac = 100.0; //the lowest accuracy in the folds
		double highest_ac = 0.0; //the highest accuracy in the folds;

		int num_pos = 0;
		int num_neg = 0;
		int num_tp = 0;
		int num_fp = 0;
		int num_tn = 0;
		int num_fn = 0;

		double fP = 0.0, fN = 0.0, tP = 0.0, tN = 0.0; //false and true negative and positive percentages.
		double accuracyTotal = 0;
		int numVal = 25;
		// Create NN using my personal class for convienience:
		for (int i = 0; i < numVal; ++i)
		{
			netCreator NC(NFL, j, i, 0.20);
			accuracyTotal += NC.accPer;
			fP += NC.falsePositives;
			fN += NC.falseNegatives;
			tP += NC.truePositives;
			tN += NC.trueNegatives;
			num_pos += NC.num_pos;
			num_neg += NC.num_neg;
			num_tp += NC.num_tp;
			num_fp += NC.num_fp;
			num_tn += NC.num_tn;
			num_fn += NC.num_fn;

			if (NC.accPer > highest_ac)
			{
				highest_ac = NC.accPer;
			}
			if (NC.accPer < lowest_ac)
			{
				lowest_ac = NC.accPer;
			}
		}
		std::wcout << std::endl << L"== Total true positives: " << (tP / static_cast<double>(numVal)) << std::endl;
		std::wcout << std::endl << L"== Total true negatives: " << (tN / static_cast<double>(numVal)) << std::endl;
		std::wcout << std::endl << L"== Total false positives:" << (fP / static_cast<double>(numVal)) << std::endl;
		std::wcout << std::endl << L"== Total false negatives:" << (fN / static_cast<double>(numVal)) << std::endl;



		//resultFile << std::endl << L"== Total percentage true positives: " << std::endl << (tP / static_cast<double>(numVal)) << std::endl;
		//resultFile << std::endl << L"== Total percentage true negatives: " << std::endl << (tN / static_cast<double>(numVal)) << std::endl;
		//resultFile << std::endl << L"== Total percentage false positives:" << std::endl << (fP / static_cast<double>(numVal)) << std::endl;
		//resultFile << std::endl << L"== Total percentage false negatives:" << std::endl << (fN / static_cast<double>(numVal)) << std::endl;


		//resultFile << std::endl << L"== Total number of true positives: " << std::endl << num_tp << std::endl;
		//resultFile << std::endl << L"== Total number of true negatives: " << std::endl << num_tn << std::endl;
		//resultFile << std::endl << L"== Total number of false positives:" << std::endl << num_fp << std::endl;
		//resultFile << std::endl << L"== Total number of false negatives:" << std::endl << num_fn << std::endl;
		//resultFile << std::endl << L"== Total number of positives:" << std::endl << num_pos << std::endl;
		//resultFile << std::endl << L"== Total number of negatives:" << std::endl << num_neg << std::endl;
		//resultFile << std::endl << L"== Total number of folds:" << std::endl << numVal << std::endl;

		theAvfN += (fN / static_cast<double>(numVal));
		theAvtN += (tN / static_cast<double>(numVal));
		theAvfP += (fP / static_cast<double>(numVal));
		theAvtP += (tP / static_cast<double>(numVal));

		Av_num_pos += num_pos;
		Av_num_neg += num_neg;
		Av_num_tp += num_tp;
		Av_num_fp += num_fp;
		Av_num_tn += num_tn;
		Av_num_fn += num_fn;

		std::wcout << std::endl << L"=== Total Accuracy: " << (accuracyTotal / static_cast<double>(numVal)) << std::endl;
		resultFile << std::endl << L"=== Total Accuracy: " << std::endl << (accuracyTotal / static_cast<double>(numVal)) << std::endl;

		std::wcout << std::endl << L"=== Higest Accuracy: " << highest_ac << std::endl;
		resultFile << std::endl << L"=== Higest Accuracy: " << std::endl << highest_ac << std::endl;

		std::wcout << std::endl << L"=== Lowest Accuracy: " << lowest_ac << std::endl;
		resultFile << std::endl << L"=== Lowest Accuracy: " << std::endl << lowest_ac << std::endl;

		theAv += (accuracyTotal / static_cast<double>(numVal));
		std::wcout << std::endl << std::endl;
		++countAv;
	}

	std::wcout << L"  Over " << countAv << L" Groups: " << std::endl;
	std::wcout << L"~Complete true positive Average: " << (theAvtP / static_cast<double>(countAv)) << std::endl;
	std::wcout << L"~Complete true negative Average: " << (theAvtN / static_cast<double>(countAv)) << std::endl;
	std::wcout << L"~Complete false positive Average: " << (theAvfP / static_cast<double>(countAv)) << std::endl;
	std::wcout << L"~Complete false negative Average: " << (theAvfN / static_cast<double>(countAv)) << std::endl;
	std::wcout << L"~~Complete Accuracy Average: " << (theAv / static_cast<double>(countAv)) << std::endl;

	//resultFile << L"  Over " << countAv << L" Groups: " << std::endl << std::endl;
	//resultFile << L"~Complete true positive Average: " << std::endl << (theAvtP / static_cast<double>(countAv)) << std::endl;
	//resultFile << L"~Complete true negative Average: " << std::endl << (theAvtN / static_cast<double>(countAv)) << std::endl;
	//resultFile << L"~Complete false positive Average: " << std::endl << (theAvfP / static_cast<double>(countAv)) << std::endl;
	//resultFile << L"~Complete false negative Average: " << std::endl << (theAvfN / static_cast<double>(countAv)) << std::endl;
	resultFile << L"~~Complete Accuracy Average: " << std::endl << (theAv / static_cast<double>(countAv)) << std::endl;
	//
	//resultFile << std::endl << L"~Complete number of true positives: " << std::endl << Av_num_tp << std::endl;
	//resultFile << std::endl << L"~Complete number of true negatives: " << std::endl << Av_num_tn << std::endl;
	//resultFile << std::endl << L"~Complete number of false positives:" << std::endl << Av_num_fp << std::endl;
	//resultFile << std::endl << L"~Complete number of false negatives:" << std::endl << Av_num_fn << std::endl;
	//resultFile << std::endl << L"~Complete number of positives:"   << std::endl << Av_num_pos << std::endl;
	//resultFile << std::endl << L"~Complete number of negatives:"   << std::endl << Av_num_neg << std::endl;
	//resultFile << std::endl << L"~Complete number of week groups:" << std::endl << countAv << std::endl;

	resultFile.close();
}


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