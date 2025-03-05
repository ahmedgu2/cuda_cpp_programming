#include "AttentionLayer.h"
#include <iostream>

int main(){
    AttentionLayer attentionLayer(32, 256, "cuda");
    std::cout << "Done!";
}