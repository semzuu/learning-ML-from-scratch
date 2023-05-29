

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>


//Global variables
#define eps  1e-3
#define rate 1e-1
#define learn_iters 1000*10
#define max_rand 2
#define min_rand 1

//-------------------------------------------------------------------
//Macros

#define mat_at(m,i,j) (m).content[(i)*(m).cols+(j)]
#define len(a) (sizeof(a) / sizeof((a)[0]))
#define mat_print(m) MAT_PRINT(m,#m,0)
#define mat_print_pad(m,p) MAT_PRINT(m,#m,p)
#define nn_print(nn) NN_PRINT(nn,#nn)
#define nn_input(nn)  ((nn).a[0])
#define nn_output(nn) ((nn).a[(nn).count])

//-------------------------------------------------------------------
//Additional functions

float rand_float(){
    //Returns a random float
    return ((float)rand())/RAND_MAX;
}
float sigmoid(float x){
    //Sigmoid function
    return 1.f/(1.f+expf(-x));
}

//-------------------------------------------------------------------
//Matrix operations

typedef struct {
    //Structure of a matrix
    int rows, cols;
    float* content;
} Mat;

Mat mat_alloc(int rows, int cols){
    //Allocates memory for a matrix
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.content = malloc((sizeof(float))*rows*cols);
    assert(m.content != NULL);
    return m;
}

void mat_rand(Mat m){
    //Randomizes the values of a matrix
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            mat_at(m,i,j) = rand_float()*(max_rand-min_rand) + min_rand;
        }
    }
}

void mat_sum(Mat res, Mat a){
    //Sums two matrices
    assert(res.cols == a.cols);
    assert(res.rows == a.rows);
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) += mat_at(a,i,j);
        }
    }
}

void mat_mult(Mat res, float scalar){
    //Multiplies a matrix by a scalar
    for(int x = 0; x < res.rows; ++x){
        for(int y = 0; y < res.cols; ++y){
            mat_at(res,x,y) *= scalar;
        }
    }    
}

void mat_dot(Mat res, Mat a, Mat b){
    //Multiplies two matrices
    assert(res.rows == a.rows);
    assert(res.cols == b.cols);
    
    for(int i = 0; i < a.rows; ++i){
        for(int j = 0; j < b.cols; ++j){
            for(int k = 0; k < b.rows; ++k){
                mat_at(res,i,j) += mat_at(a,i,k)*mat_at(b,k,j);
            }
        }
    }
}

void mat_sig(Mat res){
    //Applies sigmoid function on all values of the matrix
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) = sigmoid(mat_at(res,i,j));
        }
    }
}

void mat_reset(Mat res){
    //Set all the values of a matrix to 0
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) = 0;
        }
    }
}

void mat_copy(Mat dst, Mat a){
    //Copies a matrix
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for(int i = 0; i < dst.rows; ++i){
        for(int j = 0; j < dst.cols; ++j){
            mat_at(dst,i,j) = mat_at(a,i,j);
        }
    }
}

Mat mat_submat(Mat m, int start_row, int start_col, int row, int col){
    //Creates a sub-matrix
    Mat sub = mat_alloc(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sub.content[i * col + j] = m.content[(start_row + i) * m.cols + (start_col + j)];
        }
    }
    return sub;
}


void MAT_PRINT(Mat m, char* name, int padding){
    //Prints the matrix
    printf("%*s%s = [\n",padding,"",name);
    for(int i = 0; i < m.rows; i++){
        printf("\t");
        for(int j = 0; j < m.cols; j++){
            printf("%*s%f\t",padding,"", mat_at(m,i,j));
        }
        printf("\n");
    }
    printf("%*s]\n",padding,"");
}

//-------------------------------------------------------------------
//NN operations

typedef struct {
    //Structure of a NN
    int count;
    Mat *a;//output not included in count
    Mat *w;
    Mat *b;
} NN;

NN nn_alloc(int *node_count, int length){
    //Allocates memory for a NN
    NN nn;
    nn.count = length-1;
    nn.a = malloc((sizeof(Mat))*(nn.count+1));
    nn.w = malloc(sizeof(Mat)*nn.count);
    nn.b = malloc(sizeof(Mat)*nn.count);
    for(int i = 0; i < nn.count; i++){
        nn.a[i] = mat_alloc(1,node_count[i]);
        nn.w[i] = mat_alloc(node_count[i],node_count[i+1]);
        nn.b[i] = mat_alloc(1,node_count[i+1]);
    }
    nn.a[nn.count] = mat_alloc(1,node_count[nn.count]);
    return nn;
}


void NN_PRINT(NN nn, char* name){
    //Prints the values of the weights and biases of an NN
    int padding = 4;
    printf("%s = [\n",name);
    for(int i = 0; i < nn.count; i++){
        char index[100];
        sprintf(index, "w[%d]",i);
        MAT_PRINT(nn.w[i],index,padding);
        sprintf(index, "b[%d]",i);
        MAT_PRINT(nn.b[i],index,padding);
    }
    printf("]\n");
}

void nn_rand(NN nn){
    //Randomizes the weights and biases of an NN
    for(int i = 0; i < nn.count; i++){
        mat_rand(nn.w[i]);
        mat_rand(nn.b[i]);
    }
}

void nn_forward(NN nn){
    //Calculates output based on input
    for(int i = 0; i < nn.count; ++i){
        mat_reset(nn.a[i+1]);
        mat_dot(nn.a[i+1],nn.a[i],nn.w[i]);
        mat_sum(nn.a[i+1],nn.b[i]);
        mat_sig(nn.a[i+1]);
    }
}

float nn_node_cost(float output, float expected_output){
    //Calculates the cost of a single output node
    //Using MSE (Mean Squared Error)
    float error = expected_output - output;
    return error*error;
}

float nn_output_cost(NN nn, Mat in, Mat expected_out){
    //Calculates the cost of the NN for a single expected output
    float cost = 0;
    mat_copy(nn_input(nn),in);
    nn_forward(nn);
    for(int i = 0; i < expected_out.cols; ++i){
        //Expected output always has 1 row
        cost += nn_node_cost(mat_at(nn_output(nn),0,i),mat_at(expected_out,0,i));
    }

    return cost;
}

float nn_cost(NN nn, Mat inputs, Mat expected_outputs){
    //Calculates the cost of the NN for all the expected outputs
    float cost_total = 0;
    for(int i = 0; i < inputs.rows; i++){
        Mat input           = mat_submat(inputs,i,0,1,inputs.cols);
        Mat expected_output = mat_submat(expected_outputs,i,0,1,expected_outputs.cols);
        cost_total += nn_output_cost(nn,input,expected_output);
    }
    cost_total /= inputs.rows;
    return cost_total;
}

float slope(NN nn, Mat inputs, Mat expected_outputs, Mat current, int x, int y){
    //Calculates the difference in cost when changing the current value by a very small value 
    //(Basically a derivative)
    float deltaOutput = nn_cost(nn,inputs,expected_outputs);
    float saved = mat_at(current,x,y);

    mat_at(current,x,y) += eps;
    deltaOutput -= nn_cost(nn,inputs,expected_outputs);
    mat_at(current,x,y) = saved;
    
    float slope = deltaOutput/eps;
    return slope;
}

Mat nn_slope(NN nn, Mat inputs, Mat expected_outputs,Mat current){
    //Calculates the gradient for each layer's weights and biases
    Mat gradient = mat_alloc(current.rows,current.cols);
    for(int x = 0; x < current.rows; ++x){
        for(int y = 0; y < current.cols; ++y){
            mat_at(gradient,x,y) = slope(nn,inputs,expected_outputs,current,x,y);
        }
    }
    return gradient;
}

void nn_gradient(NN nn, NN gradient, Mat inputs, Mat expected_outputs){
    //Updates the gradient
    assert(nn.count == gradient.count);
    for(int i = 0; i < nn.count; ++i){
        assert(nn.w[i].rows == gradient.w[i].rows);
        assert(nn.w[i].cols == gradient.w[i].cols);
        assert(nn.b[i].rows == gradient.b[i].rows);
        assert(nn.b[i].cols == gradient.b[i].cols);
        Mat ws = nn_slope(nn,inputs,expected_outputs,nn.w[i]);
        mat_copy(gradient.w[i],ws);
        Mat bs = nn_slope(nn,inputs,expected_outputs,nn.b[i]);
        mat_copy(gradient.b[i],bs);
    }
}

void nn_learn(NN nn,NN gradient, Mat inputs, Mat expected_outputs){
    //Modifying weights and biases to reduce cost
    nn_gradient(nn,gradient,inputs,expected_outputs);
    for(int i = 0; i < nn.count; ++i){
        mat_mult(gradient.w[i],rate);
        mat_sum(nn.w[i],gradient.w[i]);
        mat_mult(gradient.b[i],rate);
        mat_sum(nn.b[i],gradient.b[i]);
    }
}

void nn_test(NN nn, Mat tr_in, Mat tr_out){
    //Test the inputs in the NN
    printf("Results:\n");
    int padding = 2;
    for(int i = 0; i < tr_in.rows; ++i){
        Mat in = mat_submat(tr_in,i,0,1,tr_in.cols);
        Mat out = mat_submat(tr_out,i,0,1,tr_out.cols);
        mat_copy(nn_input(nn),in);
        nn_forward(nn);
        MAT_PRINT(in,"INPUT",0);
        MAT_PRINT(nn.a[nn.count],"OUTPUT",0);
        MAT_PRINT(out,"EXPECTED OUTPUT",0);
        printf("\n");
    }
}

//--------------------------------------------------------------------

int main(){
    
    //Setting the seed
    // srand(time(0));
    srand(69); //¯\_(ツ)_/¯
    
    //Setting the training data
    float train[] = {
        0,0, 0,
        0,1, 1,
        1,0, 1,
        1,1, 0
        
    };
    
    //Initializing the NN
    int node_count[] = {2,2,1};
    NN nn = nn_alloc(node_count,len(node_count));
    NN d = nn_alloc(node_count,len(node_count));
    nn_rand(nn);
    
    //Turning the data array into a matrix
    int data_count = len(train)/(node_count[0]+node_count[len(node_count)-1]);

    Mat tr = {.rows = data_count, .cols = (node_count[0]+node_count[len(node_count)-1]), .content = train};    
    /* an Alternative
    Mat tr = mat_alloc(data_count,(node_count[0]+node_count[len(node_count)-1]));
    tr.content = train;
    */

    //Creating expected inputs and outputs
    Mat train_in  = mat_submat(tr,0,0,tr.rows,node_count[0]);
    Mat train_out = mat_submat(tr,0,node_count[0],tr.rows,node_count[len(node_count)-1]);
    
    //Learning
    printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    for(int i = 0; i < learn_iters; ++i){
        nn_learn(nn,d,train_in,train_out);
    }
    printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    
    //Testing the NN
    printf("---------------------------------\n");
    nn_test(nn,train_in,train_out);
    
    //Printing final values
    nn_print(nn);
    
    return 0;
}