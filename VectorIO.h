#ifndef VECTORIO_HEADER
#define VECTORIO_HEADER

#include <fstream>
#include <vector>


class VectorIO {
	public:

		static void write (const std::vector<float> & a, std::fstream & b);
		static void write (const std::vector<std::vector<float>> & a, std::fstream & b);
		static void write (const std::vector<std::vector<std::vector<float>>> & a, std::fstream & b);
		static void write (const std::vector<std::vector<std::vector<std::vector<float>>>> & a, std::fstream & b);

		static void read (std::vector<float> & a, std::fstream & b);
		static void read (std::vector<std::vector<float>> & a, std::fstream & b);
		static void read (std::vector<std::vector<std::vector<float>>> & a, std::fstream & b);
		static void read (std::vector<std::vector<std::vector<std::vector<float>>>> & a, std::fstream & b);

};


#endif