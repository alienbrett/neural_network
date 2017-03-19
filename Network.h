#ifndef NETWORK_HEADER
#define NETWORK_HEADER

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>
#include "VectorIO.h"


class Network {

private:

	std::vector<std::vector<std::vector<float>>> weights;
	std::vector<std::vector<float>> bias;
	std::vector<std::vector<float>> value;
	std::vector<float> input;
	std::vector<float> desired;

	std::vector<std::vector<float>> activation;
	std::vector<std::vector<float>> error;
	std::vector<std::vector<std::vector<float>>> d_weight;

	std::vector<std::vector<float>> d_bias_tally;

	char name [5];
	unsigned int * batch_number = new unsigned int;
	unsigned int  * epoch_number = new unsigned int;
	unsigned int * mini_batch_size = new unsigned int;
	float reg_rate = 0;
	float * learning_rate = new float;
	bool cross_entropy = false;
	bool l2_regularize = false;



public:

	Network ( unsigned int n, std::vector<int> & param, unsigned int mbs, float lr, bool ce, bool reg, float rr);
	bool set_input ( std::vector<float> & i);
	void get_output (std::vector<float> & o);
	void set_desired ( const std::vector<float> & y );
	void feed_forward ();
	float compute_cost ();
	void compute_errors ();
	void compute_d_weights ();
	void increment_epoch_number ();
	void clear_d_weights_and_bias ();
	void apply_derivatives ();
	void run_mini_batch ( std::vector<std::vector<float>> & i, std::vector<std::vector<float>> & d, bool speak);
	std::string get_name ();
	void save_state ();

	static float sigmoid (float a);
	static float d_sigmoid (float a);
	static void report_vector (std::vector<float> & a);
	static float multiply_and_sum (const std::vector<float>& a, const std::vector<float>& b);
	static float transpose_multiply_and_sum (const std::vector<std::vector<float>> &a, const std::vector<float> &b, unsigned int c);
	static float random (float min, float max);
	static void random_init_vector (std::vector<float> & a, float min, float max);
	static void copy_vector ( const std::vector<float> & a, std::vector<float> b);
	static char random_char();


};

//template < typename T >
//std::string to_string (const T& n);


#endif
