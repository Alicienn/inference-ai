#include <cstdio>
#include <cuda_runtime.h>
#define LS 256
#define TW 32
#define NT (LS/TW)
#define QMAX 64
__device__ __forceinline__ unsigned h4(uint4 v){return v.x^(v.y*2654435761u)^(v.z<<7)^(v.w>>3);}

__global__ void k1(const uint4* c,unsigned n,unsigned G,unsigned c1,unsigned* s){
    const unsigned st=blockIdx.x%G, team=threadIdx.x/TW, lane=threadIdx.x%TW;
    unsigned best=0;
    for(unsigned b=st+team*G;b<n;b+=G*NT){
        unsigned a=0; for(unsigned i=lane;i<c1;i+=TW) a^=h4(c[b*c1+i]);
        a^=__shfl_down_sync(0xffffffffu,a,16u); a^=__shfl_down_sync(0xffffffffu,a,8u);
        a^=__shfl_down_sync(0xffffffffu,a,4u);  a^=__shfl_down_sync(0xffffffffu,a,2u);
        a^=__shfl_down_sync(0xffffffffu,a,1u);
        if(lane==0&&a>best) best=a;
    }
    if(threadIdx.x==0) s[blockIdx.x]=best;
}
__global__ void k2(const uint4* c,unsigned n,unsigned G,unsigned c1,unsigned q,unsigned* s){
    const unsigned st=blockIdx.x%G, team=threadIdx.x/TW, lane=threadIdx.x%TW;
    __shared__ unsigned bs[NT*QMAX], bi[NT*QMAX], thr[NT];
    for(unsigned t=lane;t<q;t+=TW){bs[team*QMAX+t]=0u;bi[team*QMAX+t]=0u;}
    if(lane==0) thr[team]=0u;
    __syncthreads();
    for(unsigned b=st+team*G;b<n;b+=G*NT){
        unsigned a=0; for(unsigned i=lane;i<c1;i+=TW) a^=h4(c[b*c1+i]);
        a^=__shfl_down_sync(0xffffffffu,a,16u); a^=__shfl_down_sync(0xffffffffu,a,8u);
        a^=__shfl_down_sync(0xffffffffu,a,4u);  a^=__shfl_down_sync(0xffffffffu,a,2u);
        a^=__shfl_down_sync(0xffffffffu,a,1u);
        if(lane==0&&a>thr[team]){
            unsigned sc=a,si=b;
            for(unsigned t=0;t<q;t++){ if(sc>bs[team*QMAX+t]){
                unsigned ts=bs[team*QMAX+t],ti=bi[team*QMAX+t];
                bs[team*QMAX+t]=sc;bi[team*QMAX+t]=si;sc=ts;si=ti;}}
            thr[team]=bs[team*QMAX+q-1];
        }
    }
    __syncthreads();
    if(threadIdx.x==0) s[blockIdx.x]=bs[0];
}

__global__ void k3(const uint4* c,unsigned n,unsigned G,unsigned c1,unsigned q,unsigned* s){
    const unsigned st=blockIdx.x%G, team=threadIdx.x/TW, lane=threadIdx.x%TW;
    __shared__ unsigned bs[NT*QMAX], bi[NT*QMAX], thr[NT], sel[QMAX];
    for(unsigned t=lane;t<q;t+=TW){bs[team*QMAX+t]=0u;bi[team*QMAX+t]=0u;}
    if(lane==0) thr[team]=0u;
    __syncthreads();
    for(unsigned b=st+team*G;b<n;b+=G*NT){
        unsigned a=0; for(unsigned i=lane;i<c1;i+=TW) a^=h4(c[b*c1+i]);
        a^=__shfl_down_sync(0xffffffffu,a,16u); a^=__shfl_down_sync(0xffffffffu,a,8u);
        a^=__shfl_down_sync(0xffffffffu,a,4u);  a^=__shfl_down_sync(0xffffffffu,a,2u);
        a^=__shfl_down_sync(0xffffffffu,a,1u);
        if(lane==0&&a>thr[team]){ unsigned sc=a,si=b;
            for(unsigned t=0;t<q;t++){ if(sc>bs[team*QMAX+t]){
                unsigned ts=bs[team*QMAX+t],ti=bi[team*QMAX+t];
                bs[team*QMAX+t]=sc;bi[team*QMAX+t]=si;sc=ts;si=ti;}}
            thr[team]=bs[team*QMAX+q-1]; }
    }
    __syncthreads();
    if(threadIdx.x==0){                       /* + FUSION FINALE */
        for(unsigned t=0;t<q;t++){
            unsigned best=0,bidx=0,bj=0;
            for(unsigned j=0;j<NT*QMAX;j++) if(bs[j]>best){best=bs[j];bidx=bi[j];bj=j;}
            sel[t]=bidx; bs[bj]=0u;
        }
    }
    __syncthreads();
    if(threadIdx.x==0) s[blockIdx.x]=sel[0];
}
__global__ void k4(const uint4* c,const uint4* f,unsigned n,unsigned G,unsigned c1,
                   unsigned c2,unsigned q,unsigned* s){
    const unsigned st=blockIdx.x%G, team=threadIdx.x/TW, lane=threadIdx.x%TW;
    __shared__ unsigned bs[NT*QMAX], bi[NT*QMAX], thr[NT], sel[QMAX], red[LS];
    for(unsigned t=lane;t<q;t+=TW){bs[team*QMAX+t]=0u;bi[team*QMAX+t]=0u;}
    if(lane==0) thr[team]=0u;
    __syncthreads();
    for(unsigned b=st+team*G;b<n;b+=G*NT){
        unsigned a=0; for(unsigned i=lane;i<c1;i+=TW) a^=h4(c[b*c1+i]);
        a^=__shfl_down_sync(0xffffffffu,a,16u); a^=__shfl_down_sync(0xffffffffu,a,8u);
        a^=__shfl_down_sync(0xffffffffu,a,4u);  a^=__shfl_down_sync(0xffffffffu,a,2u);
        a^=__shfl_down_sync(0xffffffffu,a,1u);
        if(lane==0&&a>thr[team]){ unsigned sc=a,si=b;
            for(unsigned t=0;t<q;t++){ if(sc>bs[team*QMAX+t]){
                unsigned ts=bs[team*QMAX+t],ti=bi[team*QMAX+t];
                bs[team*QMAX+t]=sc;bi[team*QMAX+t]=si;sc=ts;si=ti;}}
            thr[team]=bs[team*QMAX+q-1]; }
    }
    __syncthreads();
    if(threadIdx.x==0){
        for(unsigned t=0;t<q;t++){
            unsigned best=0,bidx=0,bj=0;
            for(unsigned j=0;j<NT*QMAX;j++) if(bs[j]>best){best=bs[j];bidx=bi[j];bj=j;}
            sel[t]=bidx; bs[bj]=0u;
        }
    }
    __syncthreads();
    unsigned a=0;                              /* + PHASE 2 */
    for(unsigned t=0;t<q;t++){
        const unsigned base=sel[t]*c2;
        for(unsigned i=threadIdx.x;i<c2;i+=LS) a^=h4(f[base+i]);
    }
    red[threadIdx.x]=a; __syncthreads();
    for(unsigned sh=LS>>1;sh;sh>>=1){ if(threadIdx.x<sh) red[threadIdx.x]^=red[threadIdx.x+sh];
        __syncthreads(); }
    if(threadIdx.x==0) s[blockIdx.x]=red[0];
}
int main(){
    size_t su4=(size_t)512<<16; uint4* d; unsigned* sk;
    cudaMalloc(&d,su4*16); cudaMalloc(&sk,1<<20); cudaMemset(d,7,su4*16);
    unsigned n=4096,G=32,c1=16,q=32;
    k1<<<16*G,LS>>>(d,n,G,c1,sk); cudaDeviceSynchronize();
    printf("k1 (shfl seul)          : %s\n", cudaGetErrorString(cudaGetLastError()));
    k2<<<16*G,LS>>>(d,n,G,c1,q,sk); cudaDeviceSynchronize();
    printf("k2 (shfl + top-q partage): %s\n", cudaGetErrorString(cudaGetLastError()));
    k3<<<16*G,LS>>>(d,n,G,c1,q,sk); cudaDeviceSynchronize();
    printf("k3 (+ fusion finale)     : %s\n", cudaGetErrorString(cudaGetLastError()));
    k4<<<16*G,LS>>>(d,d,n,G,c1,64u,q,sk); cudaDeviceSynchronize();
    printf("k4 (+ phase 2, complet)  : %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;
}
