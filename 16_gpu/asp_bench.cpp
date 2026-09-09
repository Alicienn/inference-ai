// =====================================================================================
//  ASP gather benchmark — source UNIQUE compilable en CUDA (nvcc) ET en HIP (hipcc).
//
//  Question mesurée : le gather irrégulier de la passe 2 d'ASP annule-t-il le gain
//  théorique en octets ? On compare deux motifs d'accès lisant des volumes connus :
//     PLAT : flux contigu de n*S2 octets
//     ASP  : flux contigu de n*S1 octets  +  gather de m morceaux de S2 octets
//
//  Compilation
//    NVIDIA : nvcc  -O3 -arch=sm_80 asp_bench.cpp -o asp_bench
//    AMD    : hipcc -O3 --offload-arch=gfx1152 asp_bench.cpp -o asp_bench
//
//  Portage CUDA -> HIP : `hipify-perl asp_bench.cpp` fait la conversion mécanique,
//  mais les points ci-dessous demandent une relecture humaine et sont traités ici :
//    - taille de wavefront : 32 sur NVIDIA, 32 ou 64 sur RDNA (64 sur GCN/CDNA).
//      On n'utilise AUCUNE primitive de warp (pas de __shfl_*), et la réduction passe
//      par la mémoire partagée avec __syncthreads() : le code est donc insensible à
//      la largeur du wavefront. C'est le choix qui rend le portage sûr.
//    - mémoire partagée : 48 Kio par bloc sur NVIDIA, 64 Kio de LDS sur RDNA. On
//      reste à 256 * 16 = 4 Kio, très en dessous des deux.
//    - conflits de banques : la réduction accède red[lid] et red[lid+s], motif sans
//      conflit sur les deux architectures (banques de 4 octets, accès uint4 alignés).
//    - `uint4` existe des deux côtés avec la même sémantique.
// =====================================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define GPU(x)            hip##x
  #define gpuMalloc         hipMalloc
  #define gpuMemcpy         hipMemcpy
  #define gpuFree           hipFree
  #define gpuDeviceSync     hipDeviceSynchronize
  #define gpuMemcpyHtoD     hipMemcpyHostToDevice
  #define gpuEvent_t        hipEvent_t
  #define gpuEventCreate    hipEventCreate
  #define gpuEventRecord    hipEventRecord
  #define gpuEventSync      hipEventSynchronize
  #define gpuEventElapsed   hipEventElapsedTime
  #define gpuGetDeviceProps hipGetDeviceProperties
  #define gpuDeviceProp     hipDeviceProp_t
#else
  #include <cuda_runtime.h>
  #define gpuMalloc         cudaMalloc
  #define gpuMemcpy         cudaMemcpy
  #define gpuFree           cudaFree
  #define gpuDeviceSync     cudaDeviceSynchronize
  #define gpuMemcpyHtoD     cudaMemcpyHostToDevice
  #define gpuEvent_t        cudaEvent_t
  #define gpuEventCreate    cudaEventCreate
  #define gpuEventRecord    cudaEventRecord
  #define gpuEventSync      cudaEventSynchronize
  #define gpuEventElapsed   cudaEventElapsedTime
  #define gpuGetDeviceProps cudaGetDeviceProperties
  #define gpuDeviceProp     cudaDeviceProp
#endif

#define LS 64   // work-group / block : 64 threads, multiple valide des deux cotes
// Le tampon `sink` n'existe que pour empecher le compilateur d'eliminer les lectures.
// Le balayage de granularite peut lancer des millions de blocs a 64 o par morceau, bien
// plus que de cases allouees : on masque donc l'index. Les ecritures s'aliasent, ce qui
// est sans consequence ici et evite un depassement de tampon (segfault observe sur T4).
#define SINK_SLOTS (1u << 19)
#define SINK_MASK  (SINK_SLOTS - 1u)

__device__ __forceinline__ uint4 x4(uint4 a, uint4 b) {
    return make_uint4(a.x ^ b.x, a.y ^ b.y, a.z ^ b.z, a.w ^ b.w);
}

// Lecture CONTIGUE en flux : chaque bloc balaie `span` uint4 consecutifs.
__global__ void stream_k(const uint4* __restrict__ src, unsigned span,
                         uint4* __restrict__ sink) {
    __shared__ uint4 red[LS];
    const unsigned base = blockIdx.x * span;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned i = threadIdx.x; i < span; i += LS) acc = x4(acc, src[base + i]);
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = LS >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = x4(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x & SINK_MASK] = red[0];
}

// Lecture DISPERSEE : un bloc par morceau, offsets arbitraires.
__global__ void gather_k(const uint4* __restrict__ src, const unsigned* __restrict__ off,
                         unsigned chunk, uint4* __restrict__ sink) {
    __shared__ uint4 red[LS];
    const unsigned base = off[blockIdx.x];
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (unsigned i = threadIdx.x; i < chunk; i += LS) acc = x4(acc, src[base + i]);
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned s = LS >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = x4(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) sink[blockIdx.x & SINK_MASK] = red[0];
}

static float bench_stream(const uint4* d_src, size_t bytes, uint4* d_sink, int reps) {
    const unsigned span = 16384 / 16;                 // segments de 16 Kio
    const unsigned ng = (unsigned)std::max<size_t>(bytes / 16384, 1);
    gpuEvent_t a, b; gpuEventCreate(&a); gpuEventCreate(&b);
    std::vector<float> ts;
    for (int r = 0; r < reps; ++r) {
        gpuEventRecord(a);
        stream_k<<<ng, LS>>>(d_src, span, d_sink);
        gpuEventRecord(b); gpuEventSync(b);
        float ms; gpuEventElapsed(&ms, a, b); ts.push_back(ms);
    }
    std::sort(ts.begin() + 2, ts.end());
    return ts[(ts.size() + 2) / 2];
}

static float bench_gather(const uint4* d_src, const unsigned* d_off, unsigned chunk,
                          unsigned ng, uint4* d_sink, int reps) {
    gpuEvent_t a, b; gpuEventCreate(&a); gpuEventCreate(&b);
    std::vector<float> ts;
    for (int r = 0; r < reps; ++r) {
        gpuEventRecord(a);
        gather_k<<<ng, LS>>>(d_src, d_off, chunk, d_sink);
        gpuEventRecord(b); gpuEventSync(b);
        float ms; gpuEventElapsed(&ms, a, b); ts.push_back(ms);
    }
    std::sort(ts.begin() + 2, ts.end());
    return ts[(ts.size() + 2) / 2];
}

int main() {
    gpuDeviceProp p; gpuGetDeviceProps(&p, 0);
    printf("GPU : %s | %d CU/SM | %.1f GB | bus %d bits | clock %d MHz\n",
           p.name, p.multiProcessorCount, p.totalGlobalMem / 1e9,
           p.memoryBusWidth, p.clockRate / 1000);
    printf("======================================================================================\n");

    const size_t SRC_MB = 1024;
    const size_t src_u4 = (SRC_MB << 20) / 16;
    std::vector<uint4> host(src_u4);
    std::mt19937 rng(0);
    for (auto& v : host) v = make_uint4(rng(), rng(), rng(), rng());
    uint4 *d_src, *d_sink;
    gpuMalloc(&d_src, src_u4 * 16);
    gpuMalloc(&d_sink, (size_t)SINK_SLOTS * 16);
    gpuMemcpy(d_src, host.data(), src_u4 * 16, gpuMemcpyHtoD);

    printf("%6s %4s %3s %3s %7s | %9s | %9s %7s %7s | %9s %10s\n",
           "n", "D", "r1", "r2", "m", "gain th.", "t plat ms", "t p1", "t p2",
           "GAIN REEL", "efficacite");
    printf("--------------------------------------------------------------------------------------\n");

    const int Ds[]  = {64, 128};
    const int Ns[]  = {4096, 16384};
    const int R1[]  = {2, 2, 4, 4};
    const int R2[]  = {8, 16, 16, 32};
    double eff_sum = 0; int eff_n = 0, slower = 0;

    for (int D : Ds) for (int n : Ns) for (int c = 0; c < 4; ++c) {
        const int r1 = R1[c], r2 = R2[c];
        const size_t S1 = (size_t)r1 * D * 2, S2 = (size_t)r2 * D * 2;
        const unsigned chunk2 = (unsigned)(S2 / 16);
        if (chunk2 == 0 || (size_t)n * chunk2 > src_u4) continue;
        for (int frac : {4, 8}) {
            const int m = n / frac;
            if (m < 32) continue;
            std::vector<unsigned> idx(n);
            std::iota(idx.begin(), idx.end(), 0u);
            std::shuffle(idx.begin(), idx.end(), rng);
            std::vector<unsigned> off(m);
            for (int i = 0; i < m; ++i) off[i] = idx[i] * chunk2;
            unsigned* d_off; gpuMalloc(&d_off, m * sizeof(unsigned));
            gpuMemcpy(d_off, off.data(), m * sizeof(unsigned), gpuMemcpyHtoD);

            const size_t by_flat = (size_t)n * S2, by_p1 = (size_t)n * S1,
                         by_p2 = (size_t)m * S2;
            float tf = bench_stream(d_src, by_flat, d_sink, 13);
            float t1 = bench_stream(d_src, by_p1,  d_sink, 13);
            float t2 = bench_gather(d_src, d_off, chunk2, (unsigned)m, d_sink, 13);
            gpuFree(d_off);

            double gt = (double)by_flat / (by_p1 + by_p2);
            double gr = tf / (t1 + t2);
            printf("%6d %4d %3d %3d %7d | %8.2fx | %9.4f %7.4f %7.4f | %8.2fx %9.1f%%\n",
                   n, D, r1, r2, m, gt, tf, t1, t2, gr, 100.0 * gr / gt);
            eff_sum += gr / gt; ++eff_n; if (gr < 1.0) ++slower;
        }
    }
    printf("--------------------------------------------------------------------------------------\n");
    printf("efficacite moyenne (gain reel / gain theorique) : %.1f%%\n", 100.0 * eff_sum / eff_n);
    printf("configurations plus lentes que le plat : %d / %d\n", slower, eff_n);

    // =================================================================================
    //  BALAYAGE DE GRANULARITE — le test qui discrimine le modele de seuil.
    //  On lit un volume constant, decoupe en morceaux de taille croissante, contigus
    //  puis disperses. Le point ou le ratio remonte a ~1 est le seuil C*.
    // =================================================================================
    printf("\n======================================================================\n");
    printf("BALAYAGE DE GRANULARITE : ou est le seuil C* sur ce GPU ?\n");
    printf("======================================================================\n");
    printf("%9s %10s %13s %12s %8s\n", "morceau", "nb", "contigu GB/s", "gather GB/s", "ratio");
    printf("----------------------------------------------------------------------\n");
    {
        const size_t VOL = 256ull << 20;                   // 256 Mio lus par point
        const size_t MAXCH = 1ull << 20;                   // plafond de blocs lances
        for (size_t cb = 64; cb <= (32ull << 10); cb <<= 1) {
            const unsigned chunk = (unsigned)(cb / 16);
            size_t nch = VOL / cb;
            if (nch > MAXCH) nch = MAXCH;
            if (nch < 64 || nch * chunk > src_u4) continue;
            std::vector<unsigned> oc(nch), og(nch);
            for (size_t i = 0; i < nch; ++i) oc[i] = (unsigned)(i * chunk);
            std::vector<unsigned> perm(src_u4 / chunk);
            std::iota(perm.begin(), perm.end(), 0u);
            std::shuffle(perm.begin(), perm.end(), rng);
            for (size_t i = 0; i < nch; ++i) og[i] = perm[i] * chunk;
            unsigned *d_oc, *d_og;
            gpuMalloc(&d_oc, nch * 4); gpuMalloc(&d_og, nch * 4);
            gpuMemcpy(d_oc, oc.data(), nch * 4, gpuMemcpyHtoD);
            gpuMemcpy(d_og, og.data(), nch * 4, gpuMemcpyHtoD);
            float tc = bench_gather(d_src, d_oc, chunk, (unsigned)nch, d_sink, 13);
            float tg = bench_gather(d_src, d_og, chunk, (unsigned)nch, d_sink, 13);
            gpuFree(d_oc); gpuFree(d_og);
            double v = (double)nch * cb;
            printf("%8zuo %10zu %12.1f %11.1f %8.3f\n", cb, nch,
                   v / (tc * 1e-3) / 1e9, v / (tg * 1e-3) / 1e9, tc / tg);
        }
    }
    printf("\nLe seuil C* est la plus petite taille de morceau ou le ratio depasse ~0.95.\n");
    printf("Modele MLP (loi de Little) : C* = min(6144, 3.28 * B*lambda / N_res).\n");
    printf("Predictions : T4 ~328 o | A100 ~1009 o | H100 ~1559 o | MI300X ~2321 o\n");

    gpuFree(d_src); gpuFree(d_sink);
    return 0;
}
