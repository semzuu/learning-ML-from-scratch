
#define NN_IMPLEMENTATION
#include "nn.h"
#include <stdio.h>
#include <time.h>


int main(){
    
    //Setting the seed
    srand(time(0));
    //srand(69); //¯\_(ツ)_/¯

    //Setting constants
    const int min         = 0;
    const int max         = 1;
    const float eps       = 1e-1;
    const float rate      = 1e-1;
    const int learn_iters = 1000*1;
    
    //Setting the training data
    float train[] = {
        0,0, 0,
        0,1, 1,
        1,0, 1,
        1,1, 0,
    };
    
    //Initializing the NN
    int node_count[] = {4,4,1};
    NN nn = nn_alloc(node_count,len(node_count));
    NN d = nn_alloc(node_count,len(node_count));
    nn_rand(nn,min,max);
    
    //Turning the data array into a matrix
    int data_count = len(train)/(node_count[0]+node_count[len(node_count)-1]);
    printf("data_count = %d\n",data_count);
    
    Mat tr = {.rows = data_count, .cols = (node_count[0]+node_count[len(node_count)-1]), .content = train};
    //Alternatively
    //Mat tr = mat_alloc(data_count,node_count[0]+node_count[len(node_count)-1]);
    //tr.content = train;
    
    //Creating expected inputs and outputs
    Mat train_in  = mat_submat(tr,0,0,tr.rows,node_count[0]);
    Mat train_out = mat_submat(tr,0,node_count[0],tr.rows,node_count[len(node_count)-1]);
    
    //Learning
    printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    for(int i = 0; i < learn_iters; ++i){
        nn_learn(nn,d,train_in,train_out,eps,rate);
	printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    }
    
    
    //Testing the NN
    printf("---------------------------------\n");
    nn_test(nn,train_in,train_out);
    
    //Printing final values
    nn_print(nn);
    
    return 0;
}
