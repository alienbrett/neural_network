#ifndef VECTOR_STORAGE_HEADER
#define VECTOR_STORAGE_HEADER

#include <fstream>
#include <vector>
#include <cstring>


class Vector_Storage {
	private:
		char * file_name;

	public:
		Vector_Storage (char * file);
		void write_vector(std::vector<float> & a, std::ofstream & b);


};


#endif