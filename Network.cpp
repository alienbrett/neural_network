#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include <sstream>
#include "Network.h"
#include "VectorIO.h"

#define report(x) std::cout << x << std::endl

namespace patch{
	template < typename T >
	std::string to_string (const T& n){
		std::ostringstream stm;
		stm << n;
		return stm.str();
	}
}

Network::Network ( unsigned int n, std::vector<int> & param, unsigned int mbs, float lr, bool ce, bool reg, float rr)		// where n is size of input, param is std::vector of layer sizes, mbs is mini_batch_size and lr is learning rate
	{

		*learning_rate = lr; // 0.005
		*mini_batch_size = mbs;
		*epoch_number = 0;
		*batch_number = 0;

		cross_entropy = ce || reg;
		l2_regularize = reg;
		reg_rate = rr;

		input.resize(n);
		for (unsigned int x = 0; x < 5; ++x){
			this->name[x] = Network::random_char();
		}

		weights.resize(param.size());
		d_weight.resize(param.size());

		bias.resize(param.size());
		value.resize(param.size());
		activation.resize(param.size());
		error.resize(param.size());
		d_bias_tally.resize(param.size());

		for (unsigned int x = 0; x < param.size(); ++x){

			weights[x].resize(param[x]);
			d_weight[x].resize(param[x]);
			bias[x].resize(param[x]);
			activation[x].resize(param[x]);
			value[x].resize(param[x]);
			error[x].resize(param[x]);
			d_bias_tally[x].resize(param[x]);

		}

		for (unsigned int x = 0; x < weights[0].size(); x++){
			weights[0][x].resize( n );
			d_weight[0][x].resize( n );
			this->random_init_vector(weights[0][x], -2, 2);
		}

		for (unsigned int x = 1; x < weights.size(); ++x ){
			for (unsigned int y = 0; y < weights[x].size(); ++y ){
				d_weight[x][y].resize( weights[x-1].size() );
				weights[x][y].resize( weights[x-1].size() );
				this->random_init_vector(weights[x][y], -2, 2);
			}
		}

		for (unsigned int x = 0; x < bias.size(); ++x){
			this->random_init_vector(bias[x], -2, 2);
		}

		desired.resize(value.at(value.size()-1).size());
	}

bool Network::set_input ( std::vector<float> & i)
	{
		if (i.size() != input.size()){
			return false;
		} else{
			for (unsigned int x = 0; x < input.size(); ++x){
				input[x] = i[x];
			}
			return true;
		}
	}

void Network::feed_forward ()
	{
		float tmp = 0;

		for (unsigned int y = 0; y < value[0].size(); ++y){
			activation[0][y] = Network::multiply_and_sum ( input , weights[0][y] ) + bias[0][y];
			tmp = Network::sigmoid( activation[0][y] );
			value[0][y] = tmp;
		}

		for (unsigned int x = 1; x < value.size(); ++x){
			for (unsigned int y = 0; y < value[x].size(); ++y){
				activation[x][y] = Network::multiply_and_sum ( value[x-1] , weights[x][y] ) + bias[x][y];
				tmp = Network::sigmoid( activation[x][y] );
				value[x][y] = tmp;
			}
		}
	}

void Network::get_output (std::vector<float> & o)
	{
		o.resize(value[value.size()-1].size());
		for (unsigned int x = 0; x < value[value.size()-1].size(); ++x){
			o[x] = value[value.size()-1][x];
		}
	}

void Network::set_desired ( const std::vector<float> & y )
	{
		for (unsigned int x = 0; x < y.size(); ++x){
			desired[x] = y[x];
		}
	}

float Network::compute_cost ()
	{
		float tally = 0;
		float tally_reg = 0;
		unsigned int weight_n = 0;
		if (cross_entropy){
			for (unsigned int x = 0; x < desired.size(); ++x){
				tally -= desired[x]*std::log(value[value.size()-1][x]) + (1-desired[x]) * std::log(1-value[value.size()-1][x]);
			}
		} else {
			for (unsigned int x = 0; x < desired.size(); ++x){
				tally += std::pow( value[value.size()-1][x] - desired[x] ,2) / 2;
			}
		}
		if (l2_regularize){
			for (unsigned int x = 0; x < weights.size(); ++x){
				for (unsigned int y = 0; y < weights[x].size(); ++y){
					for (unsigned int z = 0; z < weights[x][y].size(); ++z){
						tally_reg += std::pow(weights[x][y][z], 2);
						++weight_n;
					}
				}
			}
			tally += (tally_reg * reg_rate) / (2*weight_n);
		}

		return tally / desired.size();
	}

void Network::compute_errors ()
	{
		for (int l = error.size() - 1; l >=0; --l){
			for (unsigned int x = 0; x < value[l].size(); ++x){

				if (l == (int)(error.size() - 1)){
					error[l][x] = (value[value.size()-1][x] - desired[x] );
						if (!cross_entropy){
							error[l][x] *= Network::d_sigmoid(activation[l][x]);
						}
					d_bias_tally[l][x] += error[l][x];
				} else {
					error[l][x] = ( this->transpose_multiply_and_sum( weights[l+1], error[l+1], x )) * Network::d_sigmoid(activation[l][x]);
					d_bias_tally[l][x] += error[l][x];
				}
			}
		}
	}

void Network::increment_epoch_number ()
	{

		++epoch_number;
	}

void Network::compute_d_weights ()
	{

		for (unsigned int x = 0; x < weights.size(); ++x){
			for (unsigned int y = 0; y < weights[x].size(); ++y){
				for (unsigned int z = 0; z < weights[x][y].size(); ++z){
					if (x == 0){
						d_weight[0][y][z] += input[z] * error[0][y];
					} else {
						d_weight[x][y][z] += value[x-1][z] * error[x][y];
					}
				}
			}
		}
	}

void Network::clear_d_weights_and_bias ()
	{
		for (unsigned int x = 0; x < weights.size(); ++x){
			for (unsigned int y = 0; y < weights[x].size(); ++y){
				error[x][y] = 0;
				d_bias_tally[x][y] = 0;
				for (unsigned int z = 0; z < weights[x][y].size(); ++z){
					d_weight[x][y][z] = 0;
				}
			}
		}
	}

void Network::apply_derivatives ()
	{
		float reg_const = 1;
		if (l2_regularize){
			reg_const -= *learning_rate * reg_rate;
		}
		for (unsigned int x = 0; x < weights.size(); ++x){
			for (unsigned int y = 0; y < weights[x].size(); ++y){
				bias[x][y] -= d_bias_tally[x][y] * *learning_rate / *mini_batch_size;
				for (unsigned int z = 0; z < weights[x][y].size(); ++z){
					weights[x][y][z] *= reg_const;
					 weights[x][y][z] -= d_weight[x][y][z] * *learning_rate / *mini_batch_size;			//
				}
			}
		}
	}

void Network::run_mini_batch ( std::vector<std::vector<float>> & i, std::vector<std::vector<float>> & d, bool speak)
	{
		float cum_cost = 0;
		for (unsigned int x = 0; x < i.size(); ++x){

			this->set_input(i[x]);
			this->feed_forward();

			this->set_desired(d[x]);
			cum_cost += this->compute_cost();

			this->compute_errors();
			this->compute_d_weights();
		}
		this->apply_derivatives();
		this->clear_d_weights_and_bias();
		(*batch_number)++;
		if (*batch_number % 100 == 0 && speak){
			std::cout<< "batch " << *batch_number << " avg cost: " << cum_cost / *mini_batch_size << std::endl;
		}
	}

std::string Network::get_name ()
	{
		char * a = new char (6);
		for (unsigned int x = 0; x < 5; ++x){
			a[x] = this->name[x];
		}
		a[5] = '\0';
		return std::string(a);
		delete a;
	}

void Network::save_state ()
	{
		std::time_t t = std::time(0);
		struct tm * now = std::localtime( & t );

		std::string temp = (patch::to_string(now->tm_year + 1900));
		temp.append("-");
		temp.append(patch::to_string(now->tm_mon + 1));
		temp.append("-");
		temp.append(patch::to_string(now->tm_mday));
		temp.append("_");
		temp.append(patch::to_string(now->tm_hour));
		temp.append(":");
		temp.append(patch::to_string(now->tm_min));
		temp.append(":");
		temp.append(patch::to_string(now->tm_sec) );
		temp.append(".");
		temp.append(this->get_name());
		temp.append(".bin");

		std::fstream f (temp, std::ios::out | std::ios::binary);
		VectorIO::write(this->weights, f);
		VectorIO::write(this->bias, f);
		f.close();
	}




float Network::sigmoid (float a)
	{

		return 1.0/(1.0+exp(-a));
	}

float Network::d_sigmoid (float a)
	{

		return exp(a) / std::pow( (exp(a)+1) , 2);
	}

void Network::report_vector (std::vector<float> & a)
	{
		for (unsigned int x = 0; x < a.size() - 1; ++x){
			std::cout << a.at(x) << ", ";
		}
		std::cout << a.at(a.size()-1) << std::endl;
	}

float Network::multiply_and_sum (const std::vector<float>& a, const std::vector<float>& b)
	{

		float tally = 0;
		for (unsigned int x = 0; x < a.size(); ++x){
			tally += a[x] * b[x];
		}
		return tally;
	}

float Network::transpose_multiply_and_sum (const std::vector<std::vector<float>> &a, const std::vector<float> &b, unsigned int c)
	{

		float tally = 0;
		for (unsigned int x = 0; x < a.size(); ++x){
			tally += a[x][c] * b[x];
		}
		return tally;
	}

float Network::random (float min, float max)
	{

		return ((max - min) * rand())/(RAND_MAX) + min;
	}

void Network::random_init_vector (std::vector<float> & a, float min, float max)
	{

		for ( unsigned int x = 0; x < a.size(); ++x){
			a[x] = Network::random(min, max);
		}
	}

void Network::copy_vector ( const std::vector<float> & a, std::vector<float> b)
	{
		if (a.size() != b.size()){
			b.resize(a.size());
		}
		for (unsigned int x = 0; x < a.size(); ++x){
			b[x] = a[x];
		}
	}

char Network::random_char ()
	{
		unsigned int a = std::floor(Network::random (48,111));
		if ( a > 57 && a < 84){
			a+= 7;
		} else if (a == 84){
			a =  95;
		} else if (a > 84){
			a += 12;
		}
		return (char)a;
	}
