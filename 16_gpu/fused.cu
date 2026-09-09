// =====================================================================================
//  ASP : noyau FUSIONNE vs DEUX NOYAUX vs PLAT — banc CUDA, trois artefacts corriges.
//
//  (1) CHARGE EGALE. Chaque requete (batch x tetes KV) scanne TOUS les n blocs : le
//      travail du selecteur vaut Q*n*S1. Les trois chemins ont des GRILLES REELLEMENT
//      DIMENSIONNEES a Q (pas une mesure a Q=1 multipliee apres coup, ce qui multipliait
//      aussi le cout de lancement).
//  (2) COALESCENCE. En phase 1, une EQUIPE de 32 threads (un warp) lit un resume de bloc
//      ensemble. Une version anterieure faisait lire un resume entier par UN thread.
//  (3) SANS BARRIERE DANS LA BOUCLE. Reduction intra-equipe par __shfl_down_sync ; chaque
//      equipe n'ecrit que dans ses propres cases partagees. Une seule __syncthreads()
//      avant la fusion finale.
//
//  Selection STRATIFIEE ENTRELACEE : le work-group g d'une requete traite les blocs
//  {g, g+G, g+2G, ...}. Cout mesure hors GPU (12_poc/code/stratified.py) : 2,9 a 3,2
//  points relatifs de rappel en entrelace, contre 22,3 en contigu.
//
//  nvcc -O3 -arch=sm_90 fused.cu -o fused
// =====================================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cuda_runtime.h>

#define LS   256
#define TW   32
#define NT   (LS/TW)
#define QMAX 64
#define SPAN 1024

__device__ __forceinline__ unsigned h4(uint4 v){
    return v.x ^ (v.y*2654435761u) ^ (v.z<<7) ^ (v.w>>3);
}

__global__ void stream_k(const uint4* __restrict__ src, unsigned span,
                         unsigned* __restrict__ sink){
    __shared__ unsigned red[LS];
    const unsigned base = blockIdx.x * span;
    unsigned acc = 0;
    for (unsigned i = threadIdx.x; i < span; i += LS) acc ^= h4(src[base + i]);
    red[threadIdx.x] = acc; __syncthreads();
    for (unsigned s = LS>>1; s; s >>= 1){
        if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = red[0];
}

__global__ void gather_k(const uint4* __restrict__ src, const unsigned* __restrict__ off,
                         unsigned chunk, unsigned* __restrict__ sink){
    __shared__ unsigned red[LS];
    const unsigned base = off[blockIdx.x];
    unsigned acc = 0;
    for (unsigned i = threadIdx.x; i < chunk; i += LS) acc ^= h4(src[base + i]);
    red[threadIdx.x] = acc; __syncthreads();
    for (unsigned s = LS>>1; s; s >>= 1){
        if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = red[0];
}

__global__ void fused_k(const uint4* __restrict__ coarse, const uint4* __restrict__ fine,
                        unsigned n, unsigned G, unsigned c1, unsigned c2, unsigned q,
                        unsigned* __restrict__ sink){
    const unsigned strate = blockIdx.x % G;
    const unsigned team = threadIdx.x / TW, lane = threadIdx.x % TW;
    __shared__ unsigned bs[NT*QMAX], bi[NT*QMAX], thr[NT], red[LS];

    // BUG CORRIGE : initialiser TOUT le tableau partage, pas seulement t<q.
    // La fusion finale balaie les NT*QMAX entrees ; les cases non initialisees
    // contenaient des restes des noyaux precedents -> indices de bloc aberrants
    // -> acces hors limites en phase 2 (diagnostique par bissection).
    for (unsigned t = threadIdx.x; t < NT*QMAX; t += LS){ bs[t]=0u; bi[t]=0u; }
    if (lane == 0) thr[team] = 0u;
    __syncthreads();

    for (unsigned b = strate + team*G; b < n; b += G*NT){
        unsigned acc = 0;
        const unsigned base = b*c1;
        for (unsigned i = lane; i < c1; i += TW) acc ^= h4(coarse[base + i]);
        acc ^= __shfl_down_sync(0xffffffffu, acc, 16u);
        acc ^= __shfl_down_sync(0xffffffffu, acc,  8u);
        acc ^= __shfl_down_sync(0xffffffffu, acc,  4u);
        acc ^= __shfl_down_sync(0xffffffffu, acc,  2u);
        acc ^= __shfl_down_sync(0xffffffffu, acc,  1u);
        if (lane == 0 && acc > thr[team]){
            unsigned sc = acc, si = b;
            for (unsigned t = 0; t < q; t++){
                if (sc > bs[team*QMAX+t]){
                    unsigned ts=bs[team*QMAX+t], ti=bi[team*QMAX+t];
                    bs[team*QMAX+t]=sc; bi[team*QMAX+t]=si; sc=ts; si=ti;
                }
            }
            thr[team] = bs[team*QMAX + q - 1];
        }
    }
    __syncthreads();

    // La FUSION FINALE (un thread balayant q*NT*QMAX entrees) dominait tout le noyau :
    // temps mesure exactement lineaire en q et independant de la taille des resumes.
    // On la supprime en stratifiant d'un cran de plus : chaque EQUIPE gathere directement
    // son propre top-(q/NT), sans merge ni synchronisation. Cout en rappel deja borne par
    // 12_poc/code/stratified.py (strates entrelacees, quota >= 2 : moins d'1 point).
    const unsigned qt = (q + NT - 1) / NT;          // quota par equipe
    unsigned acc = 0;
    for (unsigned t = 0; t < qt; t++){
        const unsigned base = bi[team*QMAX + t] * c2;
        for (unsigned i = lane; i < c2; i += TW) acc ^= h4(fine[base + i]);
    }
    red[threadIdx.x] = acc; __syncthreads();
    for (unsigned s = LS>>1; s; s >>= 1){
        if (threadIdx.x < s) red[threadIdx.x] ^= red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x] = red[0];
}

static float med(std::vector<float> v){
    std::sort(v.begin()+2, v.end());
    return v[(v.size()+2)/2];
}

#define CK(nm) { cudaError_t e_=cudaEventSynchronize(e1); cudaError_t l_=cudaGetLastError(); \
    if(e_!=cudaSuccess||l_!=cudaSuccess){ printf("%c!! ERREUR CUDA sur %s : sync=%s launch=%s%c", \
    10, nm, cudaGetErrorString(e_), cudaGetErrorString(l_), 10); exit(2);} }

int main(int argc, char** argv){
    int Q = (argc > 1) ? atoi(argv[1]) : 16;
    int REPS = (argc > 2) ? atoi(argv[2]) : 11;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("=========================================================================================\n");
    printf("ASP FUSIONNE  --  %s | %d SM | %.0f GB | bus %d bits | Q=%d\n",
           p.name, p.multiProcessorCount, p.totalGlobalMem/1e9, p.memoryBusWidth, Q);
    printf("  grilles reelles dimensionnees a Q ; 1 lancement par chemin (2 pour le 2-noyaux)\n");
    printf("=========================================================================================\n");

    const size_t SRC_MB = 4096;
    const size_t su4 = (SRC_MB << 20) / 16;
    uint4* d_src; unsigned* d_sink;
    if (cudaMalloc(&d_src, su4*16) != cudaSuccess){ printf("cudaMalloc src echec\n"); return 1; }
    if (cudaMalloc(&d_sink, (size_t)1<<28) != cudaSuccess){ printf("cudaMalloc sink echec\n"); return 1; }
    {
        std::vector<uint4> chunk(1<<20);
        std::mt19937 r(0);
        for (auto& v : chunk) v = make_uint4(r(), r(), r(), r());
        for (size_t o = 0; o < su4; o += chunk.size())
            cudaMemcpy(d_src + o, chunk.data(),
                       std::min(chunk.size(), su4-o)*16, cudaMemcpyHostToDevice);
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("\n%6s %4s %3s %3s %6s %4s %3s | %9s %9s %9s | %8s %8s %8s\n",
           "n","D","r1","r2","m","G","q","plat ms","2noyaux","fusionne",
           "gain2k","gainfus","fus/2k");
    printf("---------------------------------------------------------------------------------------------------\n");
    double sfus=0, s2k=0; int cnt=0, lose=0;
    std::mt19937 rng(1);

    for (int D : {64, 128}) for (int n : {4096, 16384})
    for (auto pr : {std::make_pair(2,8), std::make_pair(2,16),
                    std::make_pair(4,16), std::make_pair(4,32)}){
        int r1 = pr.first, r2 = pr.second;
        size_t S1 = (size_t)r1*D*2, S2 = (size_t)r2*D*2;
        unsigned c1 = (unsigned)(S1/16), c2 = (unsigned)(S2/16);
        if (!c1 || !c2 || (size_t)n*std::max(c1,c2) > su4) continue;
        for (int frac : {4, 8}){
            int m = n/frac;
            if (m < 32) continue;
            for (unsigned G : {32u, 128u, 512u}){
            unsigned q = (unsigned)std::min(std::max(m/(int)G, 1), (int)QMAX);
            if (q < 8) continue;                 // quota par equipe qt = q/NT >= 1
            unsigned gFlat = (unsigned)((size_t)Q*n*S2/(SPAN*16));
            unsigned gP1   = (unsigned)((size_t)Q*n*S1/(SPAN*16));
            unsigned gP2   = (unsigned)Q*m;
            unsigned gFus  = (unsigned)Q*G;
            if (!gFlat || !gP1) continue;

            std::vector<unsigned> off(gP2);
            for (int qq = 0; qq < Q; qq++){
                std::vector<unsigned> idx(n);
                std::iota(idx.begin(), idx.end(), 0u);
                std::shuffle(idx.begin(), idx.end(), rng);
                for (int i = 0; i < m; i++) off[qq*m+i] = idx[i]*c2;
            }
            unsigned* d_off; cudaMalloc(&d_off, (size_t)gP2*4);
            cudaMemcpy(d_off, off.data(), (size_t)gP2*4, cudaMemcpyHostToDevice);

            std::vector<float> a,b,c,d; float ms;
            for (int i=0;i<REPS;i++){ cudaEventRecord(e0);
                stream_k<<<gFlat,LS>>>(d_src, SPAN, d_sink);
                cudaEventRecord(e1); CK("plat");
                cudaEventElapsedTime(&ms,e0,e1); a.push_back(ms); }
            for (int i=0;i<REPS;i++){ cudaEventRecord(e0);
                stream_k<<<gP1,LS>>>(d_src, SPAN, d_sink);
                cudaEventRecord(e1); CK("passe1");
                cudaEventElapsedTime(&ms,e0,e1); b.push_back(ms); }
            for (int i=0;i<REPS;i++){ cudaEventRecord(e0);
                gather_k<<<gP2,LS>>>(d_src, d_off, c2, d_sink);
                cudaEventRecord(e1); CK("passe2");
                cudaEventElapsedTime(&ms,e0,e1); c.push_back(ms); }
            for (int i=0;i<REPS;i++){ cudaEventRecord(e0);
                fused_k<<<gFus,LS>>>(d_src, d_src, n, G, c1, c2, q, d_sink);
                cudaEventRecord(e1); CK("fusionne");
                cudaEventElapsedTime(&ms,e0,e1); d.push_back(ms); }
            cudaFree(d_off);

            float tf=med(a), t1=med(b), t2=med(c), tfu=med(d);
            double g2 = tf/(t1+t2), gf = tf/tfu;
            printf("%6d %4d %3d %3d %6d %4u %3u | %9.4f %9.4f %9.4f | %7.2fx %7.2fx %7.2fx\n",
                   n,D,r1,r2,m,G,q,tf,t1+t2,tfu,g2,gf,(t1+t2)/tfu);
            s2k += g2; sfus += gf; cnt++; if (tfu > t1+t2) lose++;
            }
        }
    }
    printf("---------------------------------------------------------------------------------------------------\n");
    printf("\n  gain moyen vs plat : 2 noyaux %.2fx | fusionne %.2fx\n", s2k/cnt, sfus/cnt);
    printf("  configurations ou la fusion perd contre le 2-noyaux : %d / %d\n", lose, cnt);
    cudaFree(d_src); cudaFree(d_sink);
    return 0;
}
