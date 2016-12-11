#include "netCreator.h"

void netCreator::setGenParams()
{
	// General Parameters:
	validationPercent = .2;
	alterations = true;		//--True when running a temporary alteration of program for analysis
	testing = false;		//--True when wanting to pull from stored stats vs. api stats
	overRide = false;
	num_input = 141;		//--The number of inputs per training data
	num_output = 1;			//--The number of outputs per training data

	// Training Data Parameters:
	num_layers = 3;					//--The total number of layers including the input and the output layer
	num_neurons_hidden = 20;			//--The number of neurons in the hidden layer
	desired_error = (float) 0.00001;	//--The desired get_MSE or get_bit_fail, depending of which stop function is chosen by set_train_stop_function.
	max_epochs = 300000;			//--The maximum number of epochs the training should continue
	epochs_between_reports = 0;	//--The number of epochs between printing a status report to stdout.  A value of zero means no reports should be printed.	

	// When testing simple example:
	/*
	if (overRide)
	{
		num_input = 2; 
		num_output = 1;
	}
	*/
}

void netCreator::setNetworkParams()
{
	// Neural Network Parameters:	
	net.set_training_algorithm(FANN::TRAIN_INCREMENTAL);	//--Set the training algorithm.
	//net.set_bit_fail_limit(.35);							//--Set the bit fail limit used during training.
	net.set_learning_rate((const float) 1.0);				//--Set the learning rate.
	net.set_activation_function_output(FANN::SIGMOID);		//--Set the activation function for the output layer.
	net.randomize_weights(-.1, .1);							//--Give each connection a random weight between min_weight and max_weight
	net.set_train_error_function(FANN::ERRORFUNC_LINEAR);
	net.set_train_stop_function(FANN::STOPFUNC_MSE);
	
}

void netCreator::setImpliedParams()
{
	/*
	if (!testing)
		num_data = NFL.league.size() * (week - 4);  //the number of training data sets, 4 weeks of stats + 1 week of results. - 4 because the first 4 weeks don't make up a complete set.
	else
		num_data = 379; //------------edit
	*/
	num_data = NFL.league.size() * (week - 4);
	/*
	if (overRide)
	{
		num_data = 4;
	}
	*/
}	
//------------------------------------------------------------------------------------------------------
netCreator::netCreator(int inputWeek) : week(inputWeek), NFL(inputWeek)
{
	startUp(false);
}

void netCreator::startUp(bool NFLcopied)
{
	/*------------*/
	/*--- API: ---*/
	/*------------*/

	//Start NFL API process:
	if (!NFLcopied)
		NFL.getTheStats();

	/*-----------------------*/
	/*--- NEURAL NETWORK: ---*/
	/*-----------------------*/

	//set parameters:
	setGenParams();
	setImpliedParams();

	// Review data from league (optional):
	if (!testing)
		reviewAPIData();

	//create the training data structure:
	FANN::training_data tData;
	createTrainingData(tData);

	//create the Neural Network:
	createNeuralNetwork(tData);

	//test/validate neural network:
	validateNN();

	//run neural network:
	//getPrediction(); --needs fixing

	//destroy the structures used:
	tData.destroy_train();
	net.destroy();
}

netCreator::netCreator(FootballLeague NFLin, int in_week) : NFL(NFLin), week(in_week)
{
	startUp(true);
}

void netCreator::reviewAPIData()
{
	//NFL.displayLeague(L"DeAndre Washington");

	
	std::wofstream myfile("theApiData.txt");
	if (myfile.is_open())
	{
		NFL.printLeague(myfile);
	}
	
}

double ** netCreator::buildInput()
{
	//This function will grab 1 additional set of inputs for each player, this additional set of inputs do not have a 
	//coresponding output, this is the set we would like to predict, the sets will be seperated later on. For a run 
	//that has 7 weeks of data, this function will grab 4 sets of inputs, making the array's size: 4 * NFL.league.size(). 

	double ** trainingDataSets = new double*[NFL.league.size() * (week - 3)]; //the final set to be returned.
	/*
	if (testing) //if we are just testing, we can just use an older data set
	{
		FANN::training_data tempData;
		tempData.read_train_from_file("7weekTrainData.txt");
		for (unsigned int i = 0; i < num_data; ++i) //for every set
		{
			trainingDataSets[i] = new double[num_input];
			for (unsigned int j = 0; j < num_input; ++j)//for every stat in that set
			{
				trainingDataSets[i][j] = tempData.get_input()[i][j]; //copy the read in result values to the training data set. 
			}
		}
		return trainingDataSets;
	}
	*/
	std::deque<double> currentSet; //a queue to manage and record the offsets/overlaps. (1-4, 2-5..)

	//resize the queue to 141 (num input):
	currentSet.resize(num_input);

	//store the data in the nesecary data structure:
	int count1 = 0; //number of sets recorded.
	for (auto E : NFL.league) //for each player in the league.
	{
		//---inputs[0] = E.first; 
		int count2 = -1; //number of weeks processed for that player. -1 because week[0] doesnt count
		for (auto E2 : E.second.weeklyStats) //for each week for that player.
		{
			int count3 = 0; //number of stats processed for that week for that player.
			for (auto E3 : E2.statVal) //for each statistic for that week for that player.
			{
				//---inputs[count3] = E3;
				currentSet.push_back(E3);//insert the stat into our temporary set. 
				currentSet.pop_front(); //maintain a size of 141;
				++count3; //another statistic has been processed.
			}
			++count2; //another week has been processed.
			if (count2 >= 4) //if "inputs" has read in at least 4 weeks of data. 
			{
				double * inputs = new double[num_input]; //a temperary array to retrieve stats.
				//put the player id in the first slot:
				currentSet.pop_front();
				currentSet.push_front(E.first);
				//store the temp set into the return array:
				for (unsigned int j = 0; j < num_input; ++j)
				{
					inputs[j] = currentSet[j]; //insert all of the elements in the complete set into a 
					//temperary array b/c 2d dynamic arrays are hard.
				}
				trainingDataSets[count1] = new double[num_input];
				trainingDataSets[count1] = inputs; //store the recorded stats
				++count1; //another set recorded.
			}
		}
	}
	return trainingDataSets; //return the stored stats.
}

double ** netCreator::buildOutput()
{
	double ** TDSO = new double*[num_data]; //the final set to be returned. (trainingDataSetOutput)
	/*
	if (testing)
	{
		FANN::training_data tempData;
		tempData.read_train_from_file("7weekTrainData.txt");
		for (unsigned int i = 0; i < num_data; ++i) //for every set
		{
			TDSO[i] = new double[0]; //errror here, just created a new double with 0 size
			TDSO[i][0] = tempData.get_output()[i][0]; //copy the read in result values to the training data set. 
		}
		return TDSO;
	}
	*/
	
	int count1 = 0; //numer of outputs recorded.
	for (auto E : NFL.league) //for each player in the league
	{
		//for each week for that player starting at week 5:
		for (int i = 5; i <= week; ++i) //not i < week, since the weeklyStats index range is 1 -> week, not 0 -> week - 1.
		{
			TDSO[count1] = new double[1];
			double * tempSet = new double[1];
			//I am interested in players who will score more that 15 points:
			if (E.second.weeklyStats[i].statVal[0] > 15.0)
				tempSet[0] = 1.0;
			else
				tempSet[0] = 0.0;
			TDSO[count1] = tempSet; //TDSO[0] = p1(w1-4) score w5, [1] = p1(w2-5) score w6, [2] = p1(w3-6) score w7, [3] = p2(w1-4) score w5...
			++count1;
		}
	}
	return TDSO;
}

/*
double ** netCreator::readPredFile()
{
	/*
	std::wofstream predfile("7weekPredictionData.txt");
	if (predfile.is_open())
	{
		for (int i = 0; i < NFL.league.size(); ++i)
		{
			for (int j = 0; j < num_input; ++j)
			{ 
				int temp1;
				predfile >> 
					pInput[i][j];
			}
		}
	}

	return NULL;
}
*/

void netCreator::weedOutBadCases(FANN::training_data & T)
{
	double ** tempIn = new double* [T.length_train_data()];
	double ** tempOut = new double*[T.length_train_data()];
	double ** tempReadI = T.get_input();
	double ** tempReadO = T.get_output();
	for (unsigned int i = 0; i < T.length_train_data(); ++i) //copies the actual values, not the pointers.
	{
		tempIn[i] = new double[T.num_input_train_data()];
		tempOut[i] = new double[T.num_output_train_data()];
		for (unsigned int j = 0; j < T.num_input_train_data(); ++j)
			tempIn[i][j] = tempReadI[i][j];
		for (unsigned int j = 0; j < T.num_output_train_data(); ++j)
			tempOut[i][j] = tempReadO[i][j];
	}
	weedOutBadCases(tempIn, tempOut);

	T.set_train_data(num_data, num_input, tempIn, num_output, tempOut);
} //check for num_input

void netCreator::weedOutBadCases(double **& ins, double **& outs)
{
	std::vector<double *> tempInput;
	std::vector<double *> tempOutput;

	for (unsigned int i = 0; i < num_data; ++i)
	{
		bool keepSet = false; //we will remove any set that can't show a number
		for (unsigned int j = 1; j < num_input; ++j)//j = 1, because we dont include the player ID in our search.
		{
			if (ins[i][j] > 0.0) //if it shows a number, its a keeper
				keepSet = true;
		}

		if (alterations) //if we want to run this program with these alterations
		{
			//if (outs[i][0] < 1.0) //get rid of set that doesnt have a score higher than 1
				//keepSet = false;

			if (num_input == 1 || num_input == 2) //if there are only one or two inputs per set, it doesnt matter if one of them is zero.
				keepSet = true;

			if (tempInput.size() == 1) //if there is only one input
			{
				if (outs[i][0] < 1) //and, if the output to the set is 0
					keepSet = false; //we don't want to keep it
			}
			//if (tempInput.size() >= 2) //only use two examples
			//	keepSet = false;
			//if (i % 5 != 0) //use spaced out sets
			//	keepSet = false;
			
			if (keepSet)
				alterIds.push_back(static_cast<int>(ins[i][0]));
		}

		if (keepSet)
		{
			//record locations of the good data sets:
			tempInput.push_back(ins[i]);
			tempOutput.push_back(outs[i]);
		}
		else //only delete the actual data if it is a case we dont need:
		{
			delete[] ins[i];
			delete[] outs[i];
		}
		//reset the original pointers:
		ins[i] = NULL;
		outs[i] = NULL;
	}
	//create new dynamic arrays for the original ins and 
	//outs pointers (because it will likely be a different size):
	ins = new double*[tempInput.size()];
	outs = new double*[tempOutput.size()];

	//fill the newly created arrays with pointers to the kept data sets:
	for (unsigned int i = 0; i < tempInput.size(); ++i)
	{
		ins[i] = tempInput[i];
		outs[i] = tempOutput[i];
	}

	//update the number of data sets:
	num_data = tempInput.size();
}

void netCreator::weedOutBadStats(double **& ins, double **& outs)
{
	std::vector<std::vector<double>> tempIn; //a vector to help us remove a stat from each input set
	tempIn.resize(num_data); //we can size the main part of the vector since this method does not alter the number of sets, it only alters the number of stats in each set.
	//important: tempIn is indexed the same as the dynamic arrays, although it may not see like it.

	// Record how many cases have used each particular stat:
	std::vector<int> statUsedCount; //a vector to keep track of the count for each stat. 
	statUsedCount.resize(num_input); // the size of the vector is the number of stats we start with. 

	//determine wethere or net we'd like to keep each stat
	std::vector<double> percentUsed; //a vector to tell us how often each stat has been used.
	std::vector<bool> keepStat; //a vector to mark whether or not we'd like to keep that stat.
	keepStat.resize(num_input); //the keepStat vector will be the same size throughout

	//count the instances of each stat:
	//important: these nested for loops are in an unorthodox order, we want to loop through 
	//each set one stat at a time, therefore the stat loop is surrounding the set loop:
	for (unsigned int i = 0; i < num_input; ++i)//loop through all of the stats
	{
		statUsedCount[i] = 0; //initially, set count to zero.
		for (unsigned int j = 0; j < num_data; ++j)//loop through all of the sets looking at that stat.
		{
			if (ins[j][i] > 0) //within the input vector, if that stat was "used":
				++statUsedCount[i]; //if that stat was used, increase the count of that stat.
		}

		keepStat[i] = false; //we will get rid of all stats that we can't see as usefull.

	//**code in here to determine if the stat should be kept:**
	//=========================================================
		if ((static_cast<double>(statUsedCount[i]) / static_cast<double>(num_data)) > .50)
			keepStat[i] = true;
		//if (output[i][0])

		if (alterations)
		{
			//if (i >= 1) //only use 1 stats
			//	keepStat[i] = false;
		}

	//=========================================================
		//process of removing unwanted stats:
		for (unsigned int j = 0; j < num_data; ++j) //loop through each set
		{
			if (keepStat[i])//record the value of the good stats
			{
				tempIn[j].push_back(ins[j][i]);//reading from original dynamic array
			} //tempIn[j].size() should equal tempIn[j + 1].size()
		}
	}

	num_input = tempIn[0].size(); //update the num_input value.

	//refine the input stats:
	for (unsigned int i = 0; i < num_data; ++i)//loop through all of the data sets
	{
		//delete the original dynamic array, and create a new one with the new size:
		delete[] ins[i];
		ins[i] = new double[tempIn[0].size()];

		for (unsigned int j = 0; j < num_input; ++j)//loop through all of the stats for that set
		{
			ins[i][j] = tempIn[i][j];
		}
	}
}

void netCreator::getPrediction()
{ 
	/*
	std::wofstream predfile("7weekPredictionData.txt");
	*/
	std::wofstream myfile("thePrediction.txt");
	if (myfile.is_open())
	{
		if (alterations)
		{
			//testSimple(); //----------------------------x12
			//return; //----------------------------x12


			for (unsigned int i = 0; i < alterIds.size(); ++i)
			{
				double temp1 = static_cast<double>(alterIds[i]);
				double * temp2 = &temp1;
				double * results = net.run(setsUsed[i]);
				//if (abs((temp1 / setsUsed[i][0]) - 1) > 0.00001)
				//	throw std::exception("alterIds and setsUsed didn't Line up.");
				myfile << i << L"--" << setsUsed[i][0] << L"-" << NFL.league[alterIds[i]].playerName << L"--" << *results << std::endl;
				std::wcout << i << L"--" << setsUsed[i][0] << L"-" << NFL.league[alterIds[i]].playerName << L"--" << *results << std::endl;
			}
			
			net.save("theNN.txt");
			return;
		}
		else
		{
			myfile << L"WEEK " << week + 1 << ":" << std::endl << std::endl;
			for (unsigned int i = 0; i < NFL.league.size(); ++i) //for each player in the league
			{

				int temp1 = static_cast<int>(pInput[i][0]);
				double * results = net.run(pInput[10]);

				myfile << i << L"--" << temp1 << L"-" << NFL.league[temp1].playerName << L"--" << *results << std::endl;
				/*
				if (!testing)
				{
				if (predfile.is_open())
				{
				for (int j = 0; j < num_input; ++j)
				{
				predfile << pInput[i][j] << L" ";
				}
				predfile << std::endl;
				}
				}
				*/
			}
		}
	}
	double * results = net.run(pInput[10]);
	std::wcout << *results << std::endl;

	net.save("theNN.txt");
}

void netCreator::createTrainingData(FANN::training_data & tData)
{
	//build the input and output for each set:
	input = buildInput();
	output = buildOutput();

	//seperate "input into its training and predicting set:"
	//set some variables for easy reference:
	int theSize = NFL.league.size();
	int numPlOutpts = week - 4; //the number of player outputs, also same as the number of training sets per player
	int numSets = theSize * (week - 3); //the number of sets including the prediction sets

	double ** tInput = new double*[theSize * (week - 4)];  //training. week - 4 because we arent counting the week we want to predict
	pInput = new double*[theSize]; //predicting

	//"input" has both the training and the prediction sets in it. This loop will seperate
	//the two. The comments are based on an example with 7 weeks of available stat data:
	int countT = 0; //number of inputs into tInput
	int countP = 0; //number of inputs into pInput
	for (int i = 0; i < numSets; ++i)
	{
		for (int j = 0; j < numPlOutpts; ++j) //the first 3 items go into the training sets
		{
			tInput[countT] = input[i];
			++i;
			++countT;
		}
		pInput[countP] = input[i]; //the last item goes into the prediction set
		++countP;
	}

	if (num_data == 0)
		throw std::exception("No training set's are given to train the NN.");

	if (alterations)
	{
		//refine the inputs for each data set:
		weedOutBadStats(tInput, output);
	}

	if (num_input == 0)
		throw std::exception("Statistic Filtering Has Removed All Inputs Stats. tiny rick.");

	//refine the set of training datas:
	weedOutBadCases(tInput, output);

	//scale the stats:
	scaleStats(tInput, output);

	if (num_data == 0)
		throw std::exception("Statistic and Case Filtering Have Removed All Training Sets. tiny rick.");

	//split for validation:
	splitSets(tInput, output);

	/*
	//record the stats: (temporary) //-----------x12
	//for testing puroses:
	if (alterations)
	{
		setsUsed.resize(num_data);
		//record all of the stats to be used:
		//for every set:
		for (int i = 0; i < num_data; ++i)
		{
			setsUsed[i] = new double[num_input];
			//for every stat:
			for (int j = 0; j < num_input; ++j)
			{
				setsUsed[i][j] = tInput[i][j];
			}
		}
	}
	*/

	//create training data:
	tData.set_train_data(num_data, num_input, tInput, num_output, output);
}

void netCreator::createNeuralNetwork(FANN::training_data & tData)
{
	if (num_layers == 3)
	{
		net.create_standard(num_layers, num_input, num_neurons_hidden, num_output);
	}
	else
	{
		//create aray for layer sizes in order to avoid multiLayer bug in FANN:
		unsigned int * lHid = new unsigned int[num_layers];
		for (unsigned int i = 1; i < num_layers - 1; ++i) //skip the first and last elements in the array
			lHid[i] = num_neurons_hidden;
		lHid[0] = num_input;
		lHid[num_layers - 1] = num_output;

		//create neural network:
		net.create_standard_array(num_layers, lHid);
	}

	if (overRide)
	{
		FANN::training_data oData;
		oData.read_train_from_file("oData.txt");

		//change parameters of the network:
		setNetworkParams();

		//print the parameters of the network:
		net.print_parameters();

		net.train_on_data(oData, max_epochs, epochs_between_reports, desired_error);
		oData.save_train("theOTrainingData.txt");

		return;
	}
	

	//change parameters of the network:
	setNetworkParams();

	//print the parameters of the network:
	//net.print_parameters();

	//save the training instances for using with tests:
	if (!testing)
		tData.save_train("7weekTrainData.txt");
	
	//save the training instances for looking at:
	tData.save_train("theTrainData.txt");

	//train neural network:
	net.train_on_data(tData, max_epochs, epochs_between_reports, desired_error);
}

void netCreator::testSimple()
{
	double ** temp1;
	FANN::training_data tempDat1;
	tempDat1.read_train_from_file("oData.txt");
	temp1 = tempDat1.get_input();
	
	for (int i = 0; i < 4; ++i)
	{
		std::wcout << temp1[i][0] << temp1[i][1] << std::endl;
	}

	for (int i = 0; i < 4; ++i)
	{
		double * temp2 = temp1[i];
		double * temp3;

		temp3 = net.run(temp2);
		std::wcout << temp1[i][0] << L"--" << temp1[i][1] << std::endl << *temp3 << std::endl;
	}

	std::wcout << std::endl << std::endl;
	
	double ** t1 = new double*[4];
	for (int i = 0; i < 4; ++i)
	{
		t1[i] = new double[2];
	}
	
	t1[0][0] = 0;
	t1[0][1] = 0;
	t1[1][0] = 1;
	t1[1][1] = 0;
	t1[2][0] = 0;
	t1[2][1] = 1;
	t1[3][0] = 1;
	t1[3][1] = 1;

	double * theAnswer;
	for (int i = 0; i < 4; ++i)
	{
		theAnswer = net.run(t1[i]);
		std::wcout << t1[i][0] << L"--" << t1[i][1] << std::endl << *theAnswer << std::endl;
	}
}

void netCreator::scaleStats(double **& ins, double **& outs)
{
	std::vector<double> highestStatVal;
	highestStatVal.resize(num_input);

	//loop through all of the stats:
	for (int i = 0; i < num_input; ++i)
	{
		//for each stat, the inital value to beat will be 0.0;
		highestStatVal[i] = 0.0;
		//loop through all of the sets:
		for (int j = 0; j < num_data; ++j)
		{
			if (highestStatVal[i] < ins[j][i])
				highestStatVal[i] = ins[j][i];
		}
	}

	//loop through all of the sets:
	for (int i = 0; i < num_data; ++i)
	{
		//loop through all of the stats:
		for (int j = 0; j < num_input; ++j)
		{
			//divide each stat by the highest of that stat:
			double temp = ins[i][j] / highestStatVal[j];
			ins[i][j] = temp;
		}
	}
}

void netCreator::splitSets(double **& ins, double **& outs)
{
	num_data_test = static_cast<int>(num_data * validationPercent);
	int num_data_train = num_data - num_data_test;

	//shuffle sets:
	std::vector<int> shuffledIndex;
	shuffledIndex.resize(num_data); // create an index in order to preserve the parralelism between input and output
	for (int i = 0; i < num_data; ++i)
	{
		shuffledIndex[i] = i;
	}
	std::random_shuffle(shuffledIndex.begin(), shuffledIndex.end());

	// Split the data
	//retrieve the testingData:
	testInput = new double*[num_data_test];
	testOutput = new double*[num_data_test];
	for (int i = 0; i < num_data_test; ++i)
	{
		//make sure not to shuffle the order of the stats, just the order of the players
		//use the shuffeled index to gaurentee the inputs and outputs stay linked.
		testInput[i] = ins[shuffledIndex[i]];
		testOutput[i] = outs[shuffledIndex[i]];
	}
	//retrieve the training data:
	//copy the original pointers remaining data:
	std::vector<double*> tempPointIns;
	std::vector<double*> tempPointOuts;
	for (int i = num_data_test; i < num_data; ++i)
	{
		tempPointIns.push_back(ins[shuffledIndex[i]]);
		tempPointOuts.push_back(outs[shuffledIndex[i]]);
	}

	//put the data back in arrays:
	//reset the original pointers:
	ins = new double*[num_data_train];
	outs = new double*[num_data_train];
	for (int i = 0; i < num_data_train; ++i)
	{
		ins[i] = tempPointIns[i];
		outs[i] = tempPointOuts[i];
	}

	//update num_data (num_train_data is already updated):
	num_data = num_data_train;
}

void netCreator::validateNN()
{
	int num_correct = 0;
	for (int i = 0; i < num_data_test; ++i) //for each testing set
	{
		double * test_output = net.run(testInput[i]);
		if (testOutput[i][0] > 0.999) //if the nominal value is (about) 1
		{
			if (*test_output > 0.5)
				++num_correct;
		}
		else
		{
			if (*test_output <= 0.5)
				++num_correct;
		}
	}

	double percent_correct = static_cast<double>(num_correct) / static_cast<double>(num_data_test);
	percent_correct *= 100.0;

	std::wcout << L"--accuracy: " << percent_correct << L"%"<< std::endl;
	accPer = percent_correct;
}