/* 
	Library for standard string manipulation operations
*/

#include <fstream> // for writing to files
#include <string> // for strings
#include <stdlib.h>
#include <math.h>
#include <iomanip> 
#include <algorithm> // needed for std::remove
#include "strManipLib.h"

using namespace std;
//===============================================
string delSpaces(string str) 
{
	str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
	return str;
}
//===============================================