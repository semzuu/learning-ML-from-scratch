#include <stdlib.h>
#include <time.h>
#include <math.h>


#define eps  1e-3
#define rate  1e-3
//------------------------
float rand_float(int d, int f){
    return ((((float)rand())/((float)(RAND_MAX/((f)-(d)))))+(d));
}

float sigmoid(float x){
    return 1/(1+exp(-x));
}

float step(float x){
    return (float)(x>0.5);
}

//-------------------------
float act(float w[],float b[],int x1,int x2){
    return sigmoid(w[0]*x1+w[1]*x2+b[0]);
}

float cost(int tr[][3],int size, float* w, float* b){
    float c = 0.f;
    for(int i=0;i<(size);++i){
        float d = act(w,b,tr[i][0],tr[i][1]) - tr[i][2];
        c+= d*d;
    }
    return c;
}

void test_values(int tr[][3],int size, float* w, float* b){
    for(int i=0;i<(size);++i){
        float y = act(w,b,tr[i][0],tr[i][1]);
        printf("%d | %d = %f\n",tr[i][0],tr[i][1],y);
    }
}

float dis(int tr[][3], int size,float* w, float* b, float* curr, int index){
    float original_cost = cost(tr,size,w,b);
    curr[index]+=eps;
    float d = original_cost-cost(tr,size,w,b);
    curr[index]-=eps;
    return d/eps;
    }

void learn(int tr[][3], int size, int iter, float* w, float* b){
    for(int i=0;i<(iter);++i){
        
        float dw1 = dis(tr,size,w,b,w,0);
        float dw2 = dis(tr,size,w,b,w,1);
        float db1 = dis(tr,size,w,b,b,0);
        /*
        printf("dis1 = %f\n",dw1);
        printf("dis2 = %f\n",dw2);
        printf("disb = %f\n",db1);
        printf("\n");
        */
        
        w[0]+= rate * dw1;
        w[1]+= rate * dw2;
        b[0]+= rate * db1;
    }
    
    
}

int main() {
    int training_data[][3] = {
        {0,0,0},
        {1,0,1},
        {0,1,1},
        {1,1,1}
    };
    const int size = sizeof(training_data)/sizeof(training_data[0]);
    
    
    // y = a1 * w1 + a2 * w2 + b
    srand(time(0));
    int min_rand = -5;
    int max_rand = 5;
    int learn_iterations = 1000*100;
    float w[] = {rand_float(min_rand,max_rand),rand_float(min_rand,max_rand)};
    float b[] = {rand_float(min_rand,max_rand)};
    printf("w1 = %f\n",w[0]);
    printf("w2 = %f\n",w[1]);
    printf("b = %f\n",b[0]);
    
    
    test_values(training_data,size,w,b);
    
    
    learn(training_data,size,learn_iterations,w,b);
    
    
    
    printf("---------------------------\n");
    test_values(training_data,size,w,b);
    
    
    
    int x1; printf("Donner X1 à tester: ");scanf("%d",&x1);
    int x2; printf("Donner X2 à tester: ");scanf("%d",&x2);
    printf("%d | %d = %f\n",x1,x2,step(act(w,b,x1,x2)));
    

    return 0;
}
