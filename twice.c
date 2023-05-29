
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float act(float w,int x){
    return w*x;
}

float cost(int tr[][2],int size, float w){
    
    float c = 0.f;
    for(int i=0,cost;i<(size);++i){
        float d = act(w,tr[i][0]) - tr[i][1];
        c+= d*d;
    }
    return c;
}

float rand_float(int d, int f){
    return (((float)rand()/(float)(RAND_MAX/f))+d);
}

void test_values(int tr[][2],int size, float w){
    for(int i=0;i<(size);++i){
        float y = act(w,tr[i][0]);
        printf("%f * %d = %f\n",w,tr[i][0],y);
    }
}

int main() {
    int training_data[][2] = {
        {0,0},
        {1,2},
        {2,4},
        {3,6},
        {4,8}
    };
    const int size = sizeof(training_data)/sizeof(training_data[0]);
    
    float eps = 1e-3;
    float rate = 1e-1;
    
    // y = a * w
    srand(69);
    float w = rand_float(0,10);
    printf("w = %f\n",w);
    
    
    test_values(training_data,size,w);
    
    
    for(int i=0; i<1000;++i){
        float d = cost(training_data,size,w+eps)-cost(training_data,size,w);
        //printf("dis = %f\n",d);
        /*if(d<0.f){
          w+=rate;
        }else if (d>0.f){
          w-=rate;
        }*/
	    w-=rate*d;
    }
    
    printf("---------------------------\n");
    test_values(training_data,size,w);
    
    int x; printf("Donner X à tester: ");scanf("%d",&x);
    printf("%f * %d = %f",w,x,act(w,x));
    

    return 0;
}