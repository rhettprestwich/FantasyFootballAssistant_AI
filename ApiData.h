#ifndef APIDATA_H
#define APIDATA_H

#include <cpprest/http_client.h>
#include <iostream>
#include <cpprest/json.h>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <stdexcept>
#include <fstream>

using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features


enum statType { sWeekPts = 0, s1 = 1, s2, s3, s4, s5, s6, s13, s14, s15, s16, s17, s18, s19, 
	s20, s21, s22, s23, s24, s25, s27, s28, s30, s31, s32, s70, s71, s74, 
	s75, s79, s83, sTeam, sSeasonPts, sSeasonProjectedPts, sWeekProjectedPts };

//statNames 1-92: (http://api.fantasy.nfl.com/v1/players/stats?format=json)
//stat index 0 is always the points for that week, will use week 2s points for training with week 1s data.
struct Stats
{
public:
	Stats(){ statName.resize(35); statVal.resize(35); }
	std::vector<std::wstring> statName;
	std::vector<double> statVal;

	void displayStats();
	void operator = (const Stats &S);


};

struct TempPlayer
{
public:
	TempPlayer(){}
	TempPlayer(int theWeek){ weekRecorded = theWeek; }
	int playerID;
	std::wstring playerName;
	int weekRecorded;
	
	void resetTempPlayer();

	Stats weekStats;
};

struct Player
{
public:
	Player(){ weeklyStats.resize(19); }
	Player(int theWeek){ weeklyStats.resize(theWeek); }
	std::wstring playerName;
	int playerID;

	//index by week
	std::vector<Stats> weeklyStats;

	void displayPlayer();
	Player& operator = (TempPlayer &TP);
	Player& operator += (TempPlayer &TP);
	void resetPlayer();
};

class FootballLeague
{
private:
	std::wstring indents;
	bool recordStat;
	std::vector<std::wstring> statsToRecord;
	int procWeek;
	int week_latest;

	std::vector<int> tempCheckedStats;
	int tempHitCount;
	std::map<std::wstring, int> statID;
	std::map <std::wstring, int> teamID;
	//int tempPlayerCount;

	TempPlayer tempPlayer; //needed since player id may not be first stat the api gives us.

	void IterateJSONValue(const json::value obj);
	pplx::task<void> RequestJSONValueAsync(std::wstring);

	void setStats(int playerID, int week, std::string, double);
	void record(std::wstring, std::wstring);
	bool statNeeded(std::wstring);
	void retrieveApiData();
	void tempCheckStats(int);
	void buildIDMap();

	int latestWeek;
	void buildTeamIDMap();


public:
	//index by playerID, returns a Player
	std::map<int, Player> league;

	FootballLeague(int); 
	FootballLeague();
	FootballLeague(FootballLeague & alt_NFL);

	void getTheStats();

	void displayLeague(std::wstring, bool);
	void displayLeague(std::wstring pN){ displayLeague(pN, false); }
	void displayLeague(){ displayLeague(L"you shouldnt see this", true); }
	void printLeague(std::wofstream &);

};

#endif
