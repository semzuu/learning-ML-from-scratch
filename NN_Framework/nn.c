

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define mat_at(m,i,j) (m).content[(i)*(m).cols+(j)]
#define len(a) (sizeof(a) / sizeof((a)[0]))
#define mat_print(m) MAT_PRINT(m,#m,0)
#define mat_print_pad(m,p) MAT_PRINT(m,#m,p)
#define nn_print(nn) NN_PRINT(nn,#nn)
#define nn_input(nn)  ((nn).a[0])
#define nn_output(nn) ((nn).a[(nn).count])


//Global variables
#define eps  1e-3
#define rate 1e-1
#define learn_iters 1000*10
#define max_rand 2
#define min_rand 1

//-------------------------------------------------------------------
//Additional functions

float rand_float(){
    return ((float)rand())/RAND_MAX;
}
float sigmoid(float x){
    return 1.f/(1.f+expf(-x));
}

//-------------------------------------------------------------------
//Matrix operations

typedef struct {
    int rows, cols;
    float* content;
} Mat;

Mat mat_alloc(int rows, int cols){
    Mat m;
    m.cols = cols;
    m.rows = rows;
    m.content = malloc((sizeof(float))*rows*cols);
    assert(m.content != NULL);
    return m;
}

void mat_rand(Mat m){
    for(int i = 0; i < m.rows; ++i){
        for(int j = 0; j < m.cols; ++j){
            mat_at(m,i,j) = rand_float()*(max_rand-min_rand) + min_rand;
        }
    }
}

void mat_sum(Mat res, Mat a){
    assert(res.cols == a.cols);
    assert(res.rows == a.rows);
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) += mat_at(a,i,j);
        }
    }
}

void mat_dot(Mat res, Mat a, Mat b){
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
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) = sigmoid(mat_at(res,i,j));
        }
    }
}

void mat_reset(Mat res){
    for(int i = 0; i < res.rows; ++i){
        for(int j = 0; j < res.cols; ++j){
            mat_at(res,i,j) = 0;
        }
    }
}

void mat_copy(Mat dst, Mat a){
    assert(dst.rows == a.rows);
    assert(dst.cols == a.cols);
    for(int i = 0; i < dst.rows; ++i){
        for(int j = 0; j < dst.cols; ++j){
            mat_at(dst,i,j) = mat_at(a,i,j);
        }
    }
}

Mat mat_submat(Mat m, int start_row, int start_col, int row, int col){
    Mat sub = mat_alloc(row, col);
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sub.content[i * col + j] = m.content[(start_row + i) * m.cols + (start_col + j)];
        }
    }
    return sub;
}


void MAT_PRINT(Mat m, char* name, int padding){
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
    int count;
    Mat *a;//output not included in count
    Mat *w;
    Mat *b;
} NN;

NN nn_alloc(int *node_count, int length){
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
    for(int i = 0; i < nn.count; i++){
        mat_rand(nn.w[i]);
        mat_rand(nn.b[i]);
    }
}

void nn_forward(NN nn){
    for(int i = 0; i < nn.count; ++i){
        mat_reset(nn.a[i+1]);
        mat_dot(nn.a[i+1],nn.a[i],nn.w[i]);
        mat_sum(nn.a[i+1],nn.b[i]);
        mat_sig(nn.a[i+1]);
    }
}

float nn_node_cost(float output, float expected_output){
    //Using MSE (Mean Squared Error)
    float error = expected_output - output;
    return error*error;
}

float nn_output_cost(NN nn, Mat in, Mat expected_out){

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
    float cost_total = 0;
    for(int i = 0; i < inputs.rows; i++){
        Mat input           = mat_submat(inputs,i,0,1,inputs.cols);
        Mat expected_output = mat_submat(expected_outputs,i,0,1,expected_outputs.cols);
        cost_total += nn_output_cost(nn,input,expected_output);
    }
    return cost_total/inputs.rows;
}

void nn_test(NN nn, Mat tr_in, Mat tr_out){
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
    srand(69);
    
    //Setting the training data
    float train[] = {
        0,0, 0,
        0,1, 1,
        1,0, 1,
        1,1, 0
        
    };
    int data_count = 4;
    
    //Initializing the NN
    int node_count[] = {2,2,1};
    NN nn = nn_alloc(node_count,len(node_count));
    NN d = nn_alloc(node_count,len(node_count));
    nn_rand(nn);
    
    //Turning the data array into a matrix
    Mat tr = mat_alloc(data_count,node_count[0]+node_count[len(node_count)-1]);
    tr.content = train;
    
    //Creating expected inputs and outputs
    Mat train_in  = mat_submat(tr,0,0,tr.rows,node_count[0]);
    Mat train_out = mat_submat(tr,0,node_count[0],tr.rows,node_count[len(node_count)-1]);
    
    // printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    // nn.w[0].content[0] += 5;
    // printf("cost = %f\n",nn_cost(nn,train_in,train_out));
    
    //Testing the NN
    //nn_test(nn,train_in,train_out);
    
    //Printing final values
    // nn_print(nn);
    
    return 0;
}