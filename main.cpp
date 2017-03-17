#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <string>
#include <chrono>
#include <fstream>

#include "Network.h"
#include "VectorIO.h"

#define KEEPTIME true							//enables or disables execution timekeeping
#define report(x) cout << x << endl 

using namespace std;

////////////////

int main (int argc, char * argv[]){
	
	#if KEEPTIME
		auto start = chrono::steady_clock::now();
	#endif

	srand( chrono::duration_cast<chrono::milliseconds>(chrono::time_point_cast<chrono::milliseconds>(start).time_since_epoch()).count() ); //gets epoch time in milliseconds

	
	////////////////////////////////














	unsigned int input_size = 1;
	unsigned int mini_batch_size = 10;
	float learning_rate = 0;
	unsigned int l = 0;
	unsigned int w = 0;
	unsigned int output_size = 0;
	unsigned int epoch_size = 0;
	bool testing = false;
	bool ce = false;
	bool enable_reg = false;
	float reg_rate = 0;

	for (unsigned int i = 1; i < (unsigned int)argc; i++) {
		if ( string(argv[i]) == "-b"){
			mini_batch_size = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-i"){
			input_size = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-r"){
			learning_rate = atof(argv[i+1]);
		} else if (string(argv[i]) == "-l"){ // length of network
			 l = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-w"){ // width of network
			w = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-o"){
			output_size = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-e"){
			epoch_size = atoi(argv[i+1]);
		} else if (string(argv[i]) == "-l2"){
			reg_rate = atof(argv[i+1]);
			enable_reg = true;
			ce = true;
		} else if (string(argv[i]) == "-t"){
			testing = true;
		} else if (string(argv[i]) == "--cross-entropy"){
			ce = true;
		}
	}

	vector<int> dimensions;
	for (unsigned int x = 0; x < l; ++x){
		dimensions.push_back(w);
	}
	dimensions.push_back(output_size);

	Network nn ( input_size, dimensions, mini_batch_size, learning_rate, ce, enable_reg, reg_rate);
	report("created network named " <<nn.get_name());
	vector<vector<float>> i;
	vector<vector<float>> o;
	i.resize(mini_batch_size);
	o.resize(mini_batch_size);
	
	if (testing){
		for (unsigned int x = 0; x < mini_batch_size; ++x){

			i[x].resize(input_size);
			o[x].resize(input_size);
			Network::random_init_vector(i[x], 0, 1);
			for (unsigned int y = 0; y < input_size; ++y){
				o[x][y] = pow( i[x][y] , 3) ;
			}
		}
	}

	report("about to run batches...");
	for (unsigned int x = 0; x < epoch_size; ++x){
		nn.run_mini_batch(i, o, true);
	}
	nn.save_state();
	

	////////////////////////////////

	#if KEEPTIME
		auto end = chrono::steady_clock::now();
		chrono::duration<float> elapsed = chrono::duration_cast<chrono::nanoseconds>(end - start);
		auto millisec = elapsed.count() * 1000;
		auto secs = floor(millisec/1000);
		auto mins = floor(secs/60);
		auto hours = floor(mins/60);

		cout << "\nExecution time: ";

		if (hours){
			cout << hours << "hr, ";
		}
		if (mins){
			cout << mins << "m, ";
		}
		if (secs){
			cout << secs << "s, ";
		}
		if (millisec){
			cout << fmod(millisec,1000 )<< "ms" << endl;
		}
	#endif

	return 0;
}

////////////////////////////////////////////////////////////////