#include <fstream>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include "VectorIO.h"

void VectorIO::write(const std::vector<float> & a, std::fstream & b)
    {
    	unsigned int l = a.size();
        b.write( (char*)&l, sizeof(l) );
        b.write( (const char*) & a[0], l * sizeof(float) );
    }

void VectorIO::write(const std::vector<std::vector<float>> & a, std::fstream & b)
    {
        unsigned int l = a.size();

        b.write( (char*)&l, sizeof(l) );
        for (unsigned int x = 0; x < l; ++x)
            VectorIO::write(a[x], b);
    }

void VectorIO::write(const std::vector<std::vector<std::vector<float>>> & a, std::fstream & b)
    {
        unsigned int l = a.size();

        b.write( (char*)&l, sizeof(l) );
        for (unsigned int x = 0; x < l; ++x)
            VectorIO::write(a[x], b);
    }

void VectorIO::write(const std::vector<std::vector<std::vector<std::vector<float>>>> & a, std::fstream & b)
    {
        unsigned int l = a.size();

        b.write( (char*)&l, sizeof(l) );
        for (unsigned int x = 0; x < l; ++x)
            VectorIO::write(a[x], b);
    }

void VectorIO::read(std::vector<float>& a, std::fstream & b)
    {
        unsigned int l = 0;
        b.read( (char*)&l, sizeof(l) );
        a.resize(l);
        //std::cout << l << std::endl;
        if( l > 0 )
            b.read( (char*)&a[0], l * sizeof(float) );
    }


void VectorIO::read(std::vector<std::vector<float>> & a, std::fstream & b)
    {
        unsigned int l = 0;
        b.read( (char*)&l, sizeof(l) );
        a.resize(l);
        //std::cout << l << " vectors being read..." << std::endl;
        if( l > 0 ){
            for (unsigned int x = 0; x < l; ++x){
                VectorIO::read(a[x], b);
            }
        }
        //std::cout << "done!" << std::endl;
    }

void VectorIO::read(std::vector<std::vector<std::vector<float>>> & a, std::fstream & b)
    {
        unsigned int l = 0;
        b.read( (char*)&l, sizeof(l) );
        a.resize(l);
        //std::cout << l << " vectors being read..." << std::endl;
        if( l > 0 ){
            for (unsigned int x = 0; x < l; ++x){
                VectorIO::read(a[x], b);
            }
        }
        //std::cout << "done!" << std::endl;
    }

void VectorIO::read(std::vector<std::vector<std::vector<std::vector<float>>>> & a, std::fstream & b)
    {
        unsigned int l = 0;
        b.read( (char*)&l, sizeof(l) );
        a.resize(l);
        //std::cout << l << " vectors being read..." << std::endl;
        if( l > 0 ){
            for (unsigned int x = 0; x < l; ++x){
                VectorIO::read(a[x], b);
            }
        }
        //std::cout << "done!" << std::endl;
    }