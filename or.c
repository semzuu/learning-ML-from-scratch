#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


#define eps  1e-3
#define rate  1e-1

float sigmoid(float x){
    return 1/(1+exp(-x));
}

float step(float x){
    return x>0.f;
}

float act(float w[],int x1,int x2){
    return w[0]*x1+w[1]*x2+w[2];
}

float cost(int tr[][3],int size, float* w){
    
    float c = 0.f;
    for(int i=0;i<(size);++i){
        float d = act(w,tr[i][0],tr[i][1]) - tr[i][2];
        c+= d*d;
    }
    return c;
}

float rand_float(int d, int f){
    return (((float)rand()/(float)(RAND_MAX/f))+d);
}

void test_values(int tr[][3],int size, float w[]){
    for(int i=0;i<(size);++i){
        float y = act(w,tr[i][0],tr[i][1]);
        printf("%d | %d = %f\n",tr[i][0],tr[i][1],y);
    }
}

void add_eps(float w[],int index){
    w[index]+=eps;
}

/*float* add_eps(float w[], int index) {
    float* w_modified = malloc(3 * sizeof(float));
    for (int i = 0; i < 3; ++i) {
        w_modified[i] = w[i];
    }
    w_modified[index] += eps;
    return w_modified;
}*/
void sub_eps(float w[],int index){
    *(w+index)-=eps;
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
    srand(69);
    int max_rand = 10;
    float w[] = {rand_float(0,max_rand),rand_float(0,max_rand),rand_float(0,max_rand)};
    printf("w1 = %f\n",w[0]);
    printf("w2 = %f\n",w[1]);
    printf("b = %f\n",w[2]);
    
    
    test_values(training_data,size,w);
    
    
    for(int i=0; i<1000*100;++i){
        float original_cost = cost(training_data,size,w);
        add_eps(w,0);
        float d1 = cost(training_data,size,w)-original_cost;
        sub_eps(w,0);
        
        add_eps(w,1);
        float d2 = cost(training_data,size,w)-original_cost;
        sub_eps(w,1);
        
        add_eps(w,2);
        float db = cost(training_data,size,w)-original_cost;
        sub_eps(w,2);
        /*
        printf("dis1 = %f\n",d1);
        printf("dis2 = %f\n",d2);
        printf("disb = %f\n",db);
        printf("\n");
        */
        w[0]-= rate * d1;
        w[1]-= rate * d2;
        w[2]-= rate * db;
    }
    printf("---------------------------\n");
    test_values(training_data,size,w);
    
    
    
    int x1; printf("Donner X1 à tester: ");scanf("%d",&x1);
    int x2; printf("Donner X2 à tester: ");scanf("%d",&x2);
    printf("%d | %d = %f\n",x1,x2,round(act(w,x1,x2)));
    

    return 0;
}