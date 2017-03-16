#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include "Vector_Storage.h"

Vector_Storage::Vector_Storage(char * file){
	file = new char[1024];
	strcpy(file_name, file);
}

void Vector_Storage::write_vector(std::vector<float> & a, std::ofstream & b){
	unsigned int l = a.size();
    b.write( (char*)&l, sizeof(l) );
    b.write( (const char*) & a[0], l * sizeof(float) );
}
/*

template <class T>
void WriteTrivial( std::ostream& s, const std::vector<T>& data )
{
    unsigned int len = data.size();
    s.write( (char*)&len, sizeof(len) );
    s.write( (const char*)&data[0], len * sizeof(T) );
}

// This reads a vector of trivial data types.
template <class T>
void ReadTrivial( std::istream& s, std::vector<T>& data )
{
    unsigned int len = 0;
    s.read( (char*)&len, sizeof(len) );
    data.resize(len);
    if( len > 0 ) s.read( (char*)&data[0], len * sizeof(T) );
}

*/