#include <iostream>
#include <random>

// Utils
void initVector(float *vector, const int length, unsigned int seed = 42)
{
    std::mt19937 gen(seed);
    // std::uniform_int_distribution<int> dist(1, 255);
    std::normal_distribution<float> dist;
    for (int i = 0; i < length; ++i)
    {
        vector[i] = (float)dist(gen);
    }
}

void printVector(float *vector, const int length)
{
    for (int i = 0; i < length; ++i)
    {
        std::cout << vector[i] << " ";
    }
    std::cout << std::endl;
}