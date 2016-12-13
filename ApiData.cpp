#include "ApiData.h"
#include <assert.h>

void Stats::displayStats()
{
	int count1 = 0;
	for (auto E : statName)
	{
		std::wcout << L"  " << E << L": " << statVal[count1] << std::endl;
		++count1;
	}
}
void Stats::operator = (const Stats &S)
{
	this->statName = S.statName;
	this->statVal = S.statVal;
}

void TempPlayer::resetTempPlayer()
{ 
	playerID = 0; 
	playerName.clear(); 
	weekRecorded = 0;
	weekStats.statName.clear();
	weekStats.statVal.clear();
	weekStats.statName.resize(35);
	weekStats.statVal.resize(35);
}

void Player::displayPlayer()
{
	int count2 = 0;
	std::wcout << L"---------------" << playerName << L"---------------" << std::endl;
	for (auto E : weeklyStats)
	{
		if (!count2 == 0)
		{
			std::wcout << L"---------Week " << count2 << L"---------" << std::endl;
			E.displayStats();
		}
		++count2;
	}
	std::wcout << std::endl;
}
Player& Player::operator = (TempPlayer &TP)
{
	this->playerID = TP.playerID;
	this->playerName = TP.playerName;
	this->weeklyStats[TP.weekRecorded] = TP.weekStats;
	return *this;
}
Player& Player::operator +=(TempPlayer &TP)
{
	if (this->playerID == TP.playerID)
	{
		this->weeklyStats[TP.weekRecorded] = TP.weekStats;
	}
	else
		assert((this->playerID == TP.playerID) && "Cannot use \"+=\" with two different players (tiny rick)");
	return *this;
}

void Player::resetPlayer()
{
	playerID = 0;
	playerName.clear();

	for (auto EP : weeklyStats)
	{
		EP.statName.clear();
		EP.statVal.clear();
	}
}

void FootballLeague::displayLeague(std::wstring theP, bool dispAll)
{
	std::wcout << L"-------------------League-------------------" << std::endl;
	for (std::map<int, Player>::iterator it = league.begin(); it != league.end(); ++it)
	{
		if (it->second.playerName == theP || dispAll)
		{
			it->second.displayPlayer();
			std::wcout << std::endl;
		}
	}
	std::wcout << std::endl;
}

void FootballLeague::buildIDMap()
{
	std::vector<std::wstring> statNames = { L"weekPts", L"1", L"2", L"3", L"4", L"5", L"6",
		L"13",L"14", L"15", L"16", L"17", L"18", L"19", L"20", L"21", L"22",
		L"23", L"24", L"25", L"27", L"28", L"30", L"31", L"32", L"70", L"71",
		L"74", L"75", L"79", L"83", L"weekProjectedPts", L"seasonProjectedPts", L"seasonPts", L"teamAbbr"};
	int counter = 0;
	for (auto E : statNames)
	{
		statID[E] = counter;
		++counter;
	}
}

void FootballLeague::buildTeamIDMap()
{
	std::vector<std::wstring> teamNames = { L"ARI", L"ATL", L"BAL", L"BUF", L"CAR", L"CHI",
		L"CIN", L"CLE", L"DAL", L"DEN", L"DET", L"GB", L"TEN", L"HOU", L"IND", L"JAX", 
		L"KC", L"STL", L"OAK", L"MIA", L"MIN", L"NE", L"NO", L"NYG", L"NYJ", L"PHI", L"PIT",
		L"SD", L"SF", L"SEA", L"TB", L"WAS"};
	int counter = 0;
	for (auto E : teamNames)
	{
		teamID[E] = counter;
		++counter;
	}
}

FootballLeague::FootballLeague(int theWeek)
{
	week_latest = theWeek;
}

void FootballLeague::getTheStats()
{
	latestWeek = week_latest;
	buildIDMap();
	buildTeamIDMap();
	recordStat = false;
	statsToRecord = { L"id", L"name", L"seasonPts", L"seasonProjectedPts", L"weekPts",
		L"weekProjectedPts", L"teamAbbr", L"id" };

	std::wcout << std::endl << L"	~Requesting Stats From NFL.com...";
	retrieveApiData();
	std::wcout << L"|" << std::endl << std::endl;

	/*
	std::sort(tempCheckedStats.begin(), tempCheckedStats.end());
	int theCount = 0;
	for (auto i : tempCheckedStats)
	{
	std::wcout << L"L\"" << i << L"\", ";
	++theCount;
	}
	std::wcout << std::endl << L"Total number of stats: " << theCount << std::endl;
	*/
}

FootballLeague::FootballLeague()
{
	/*FootballLeague(19);*/
}

FootballLeague::FootballLeague(FootballLeague & alt_NFL) //only copies the league since that's all we use.
{
	this->league = alt_NFL.league;
	this->recordStat = alt_NFL.recordStat;
	this->indents = alt_NFL.indents;
	this->statsToRecord = alt_NFL.statsToRecord;
	this->procWeek = alt_NFL.procWeek;
	this->week_latest = alt_NFL.week_latest;
	this->tempCheckedStats = alt_NFL.tempCheckedStats;
	this->tempHitCount = alt_NFL.tempHitCount;
	this->statID = alt_NFL.statID;
	//not everything is copied
}

void FootballLeague::retrieveApiData()
{
	std::wcout << std::endl;
	std::wstring callAddress, weekNum;
	for (int i = 1; i <= latestWeek; ++i)
	{
		tempHitCount = 1;
		procWeek = i;
		weekNum = std::to_wstring(i);
		callAddress = L"http://api.fantasy.nfl.com/v1/players/stats?position=RB&statType=weekStats&format=json&week=";
		callAddress += weekNum;
		if (procWeek != 1)
			std::wcout << L"|";
		std::wcout << std::endl << L"Obtaining Data From Week " << i << L" ";
		//tempPlayerCount = 0;
		RequestJSONValueAsync(callAddress).wait();

		//std::wcout << std::endl << L"Number of RBs" << tempPlayerCount << std::endl;
	}
}

// Demonstrates how to iterate over a JSON object. 
void FootballLeague::IterateJSONValue(const json::value obj)
{
	// Loop over each element in the object. 
	if (!obj.is_null())
	{
		for (auto iter = obj.as_object().cbegin(); iter != obj.as_object().cend(); ++iter)
		{
			// Make sure to get the value as const reference otherwise you will end up copying 
			// the whole JSON value recursively which can be expensive if it is a nested object. 
			const string_t &key = iter->first;
			const json::value &value = iter->second;

			// Perform actions here to process each string and value in the JSON object...
			if (value.is_object() || value.is_array())
			{
				// We have an object with children or an array
				// Loop over each element in the object by calling DisplayJSONValue
				//---std::wcout << indents << L"Parent: " << key << std::endl;
				
				if (key == L"stats")
					recordStat = true;

				if (value.is_array()) //(ie: key=players value=array of player objects)
				{
					//---std::wcout << indents << L"[" << std::endl;
					indents += L"     ";
					int arrSize = value.as_array().size();
					for (int i = 0; i < arrSize; ++i)
					{
						//if its an array, itterate through all of the objects in the array
						
						//---std::wcout << std::endl << indents << L"{" << std::endl;
						indents += L"     ";
						json::array::size_type ind = i;
						
						
						tempPlayer.resetTempPlayer();
						tempPlayer.weekRecorded = procWeek;

						IterateJSONValue(value.as_array().at(ind)); //won't work for an array of arrays, only for an array of objects

						//Player toDisplay;
						//toDisplay.weeklyStats.resize(procWeek + 1);
						//toDisplay = tempPlayer;
						//toDisplay.displayPlayer();
						//++tempPlayerCount;
						if (league.count(tempPlayer.playerID))
						{
							league[tempPlayer.playerID] += tempPlayer;
						}
						else
						{
							league[tempPlayer.playerID].weeklyStats.resize(latestWeek + 1);
							league[tempPlayer.playerID] = tempPlayer;
						}
						/*
						if (procWeek == latestWeek)
						{
							if (tempPlayer.playerName == L"LeSean McCoy")
								league[tempPlayer.playerID].displayPlayer();
						}

						for (int j = 0; j < 5; ++j)
							indents.pop_back();
						//---std::wcout << indents << L"}" << std::endl;
						*/
					}
					for (int j = 0; j < 5; ++j)
						indents.pop_back();
					//---std::wcout << std::endl << indents << L"]" << std::endl;
					recordStat = false;
				}
				else //only gets hit if the obj passed in is an object with a key (ie: stats)
				{
					//---std::wcout << indents << L"{" << std::endl;
					indents += L"     ";
					IterateJSONValue(value);
					for (int j = 0; j < 5; ++j)
						indents.pop_back();
					//---std::wcout << indents << L"}" << std::endl;
					recordStat = false;
				}
				//---std::wcout << indents << L"End of Parent: " << key << std::endl;
			}
			else
			{
				// Always display the value as a string
				//---std::wcout << indents << L"Key: " << key << L", Value: " << value.serialize();
				if (statNeeded(key))
					record(key, value.serialize());
				//---std::wcout << std::endl;
			}
		}
	}
}

// Retrieves a JSON value from an HTTP request.
pplx::task<void> FootballLeague::RequestJSONValueAsync(std::wstring api_address)
{
	// TODO: To successfully use this example, you must perform the request  
	// against a server that provides JSON data.  
	// This example fails because the returned Content-Type is text/html and not application/json.
	http_client client(api_address);

	//uri_builder builder(L"/players/weekstats");
	return client.request(methods::GET).then([](http_response response) -> pplx::task<json::value>
	{
		if (response.status_code() == status_codes::OK)
		{
			return response.extract_json();
		}

		// Handle error cases, for now return empty json value... 
		return pplx::task_from_result(json::value());
	})
		.then([this](pplx::task<json::value> previousTask)
	{
		try
		{
			const json::value& v = previousTask.get();
			// Perform actions here to process the JSON value...
			IterateJSONValue(v);
		}
		catch (const http_exception& e)
		{
			// Print error.
			std::wostringstream ss;
			ss << e.what() << std::endl;
			std::wcout << ss.str();
		}
	});

	/* Output:
	Content-Type must be application/json to extract (is: text/html)
	*/
}

void FootballLeague::tempCheckStats(int theStat)
{
	bool newStat = true;
	if (tempCheckedStats.size() == 0)
	{
		tempCheckedStats.push_back(theStat);
	}
	else
	{
		for (unsigned int i = 0; i < tempCheckedStats.size(); ++i)
		{
			if (theStat == tempCheckedStats[i])
				newStat = false;
		}
		if (newStat)
		{
			tempCheckedStats.push_back(theStat);
			//---------std::wcout << L"NEW STAT: " << theStat << std::endl;
		}
	}
}

void FootballLeague::record(std::wstring theName, std::wstring theValue)
{
	if (tempHitCount % 45 == 0)
		std::wcout << L"-";
	++tempHitCount;
	//removing quotation marks:
	std::wstring N = theName;
	std::wstring V = theValue;
	if (V[0] == L'\"')
	{
		V.pop_back();
		V.erase(0, 1);
	}

	//---std::wcout << " <--- (recorded)";

	if (N == L"name")
	{
		tempPlayer.playerName = V;
		return;
	}
	else if (N == L"id")
	{
		int tempVal = std::stoi(V);
		tempPlayer.playerID = tempVal;
	}
	else if (N <= L"93")//meaning it is a stat with an index
		tempCheckStats(std::stoi(N));
	else if (N == L"teamAbbr")
	{
		tempPlayer.weekStats.statName[statID[N]] = N;
		tempPlayer.weekStats.statVal[statID[N]] = teamID[V];
		return;
	}
	tempPlayer.weekStats.statName[statID[N]] = N;
	tempPlayer.weekStats.statVal[statID[N]] = std::stod(V);

}

bool FootballLeague::statNeeded(std::wstring theName)
{
	for (auto E : statsToRecord)
	{
		if (E == theName)
			return true;
	}
	if (recordStat)
		return true;

	return false;
}

void FootballLeague::printLeague(std::wofstream & theFile)
{
	 theFile << L"-------------------League-------------------" << std::endl;
	for (std::map<int, Player>::iterator it = league.begin(); it != league.end(); ++it)
	{
		int count2 = 0;

		theFile << L"---------------" << it->second.playerName << L"---------------" << std::endl;
		for (auto E : it->second.weeklyStats)
		{
			if (!count2 == 0)
			{
				theFile << L"---------Week " << count2 << L"---------" << std::endl;
				int count1 = 0;
				for (auto E2 : E.statName)
				{
					theFile << L"  " << E2 << L": " << E.statVal[count1] << std::endl;
					++count1;
				}
			}
			++count2;
		}
		theFile << std::endl;
		theFile << std::endl;
	}
	theFile << std::endl;
	

	
	




}