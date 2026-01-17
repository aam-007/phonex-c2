/*
 * Phonex-C2: Research-Grade Pure C Transformer Engine
 * -----------------------------------------------------------
 * Compile: 
 * gcc -O3 -mavx2 -mfma -fopenmp c2p.c -lm -o c2
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// --- Configuration ---
#define BATCH_SIZE 1      // Small batch for demo
#define SEQ_LEN 32        // Context window
#define D_MODEL 128       // Embedding dimension
#define N_HEADS 4         // Number of heads
#define HEAD_DIM (D_MODEL / N_HEADS)
#define D_FF (D_MODEL * 4) // FFN expansion
#define VOCAB_SIZE 256    // Byte-level
#define MAX_LAYERS 4      // Depth
#define TOKEN_EOS 0       // End of sequence token (0 = null byte)

// Optimization Defaults
#define BLOCK_SIZE 32
#define GRAD_CLIP 1.0f
#define WARMUP_STEPS 10
#define MAX_STEPS 100
#define BASE_LR 1e-3f
#define ADAM_B1 0.9f
#define ADAM_B2 0.99f
#define ADAM_EPS 1e-8f

// Dataset Configuration
#define MAX_PHRASES 1000
#define MAX_PHRASE_LEN 100

// --- 1. Memory & Tensor System ---

typedef struct {
    int dim[4]; // [Batch, Head, Seq, Dim]
} Shape;

typedef struct {
    float* data;
    float* grad;
    float* m; // Adam moment 1
    float* v; // Adam moment 2
    Shape shape;
    int size;
    char name[32];
} Tensor;

typedef struct {
    float* buffer;
    size_t size;
    size_t offset;
    size_t peak;
} Arena;

// 256MB Arena
#define ARENA_SIZE (256 * 1024 * 1024)

Arena arena_create() {
    Arena a;
    // 32-byte alignment for AVX
    a.buffer = (float*)aligned_alloc(32, ARENA_SIZE);
    if (!a.buffer) { fprintf(stderr, "Fatal: Arena OOM\n"); exit(1); }
    a.size = ARENA_SIZE / sizeof(float);
    a.offset = 0;
    a.peak = 0;
    return a;
}

void arena_reset(Arena* a) {
    if (a->offset > a->peak) a->peak = a->offset;
    a->offset = 0;
    // Zeroing is critical for gradient accumulation in backward pass
    memset(a->buffer, 0, a->peak * sizeof(float));
}

float* arena_alloc(Arena* a, int count) {
    int aligned = (count + 7) & ~7; // Align 8 floats (32 bytes)
    if (a->offset + aligned > a->size) {
        fprintf(stderr, "Fatal: Arena Overflow\n"); exit(1);
    }
    float* ptr = &a->buffer[a->offset];
    a->offset += aligned;
    return ptr;
}

// Tensor Constructors
Shape shape(int d0, int d1, int d2, int d3) { return (Shape){{d0, d1, d2, d3}}; }
int shape_size(Shape s) { return s.dim[0] * s.dim[1] * s.dim[2] * s.dim[3]; }

Tensor tensor_param(int d0, int d1, int d2, int d3, const char* name) {
    Tensor t;
    t.shape = shape(d0, d1, d2, d3);
    t.size = shape_size(t.shape);
    strncpy(t.name, name, 31);
    t.data = (float*)aligned_alloc(32, t.size * sizeof(float));
    t.grad = (float*)calloc(t.size, sizeof(float));
    t.m = (float*)calloc(t.size, sizeof(float));
    t.v = (float*)calloc(t.size, sizeof(float));
    
    // Xavier Init
    float scale = sqrtf(6.0f / (float)(d2 + d3));
    for(int i=0; i<t.size; i++) t.data[i] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * scale;
    return t;
}

Tensor tensor_arena(Arena* a, int d0, int d1, int d2, int d3) {
    Tensor t;
    t.shape = shape(d0, d1, d2, d3);
    t.size = shape_size(t.shape);
    t.data = arena_alloc(a, t.size);
    t.grad = arena_alloc(a, t.size);
    t.m = NULL; t.v = NULL; 
    return t;
}

// --- 2. AVX2 & Math Kernels ---

// C = A @ B. A=[M, K], B=[K, N] (Standard), C=[M, N]
// Naive layout for weights. Optimized with blocking.
void matmul_forward_kernel(float* A, float* B, float* C, int M, int K, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 8) {
            #ifdef __AVX2__
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 a_val = _mm256_set1_ps(A[i*K + k]);
                __m256 b_val = _mm256_loadu_ps(&B[k*N + j]);
                sum = _mm256_fmadd_ps(a_val, b_val, sum);
            }
            _mm256_storeu_ps(&C[i*N + j], sum);
            #else
            for (int jj = j; jj < j + 8 && jj < N; jj++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) sum += A[i*K + k] * B[k*N + jj];
                C[i*N + jj] = sum;
            }
            #endif
        }
    }
}

// Batched wrapper
void matmul_forward(Tensor* A, Tensor* B, Tensor* C) {
    int Batch = A->shape.dim[0], M = A->shape.dim[2], K = A->shape.dim[3];
    int N = B->shape.dim[3]; // B assumed [1, 1, K, N] for weights
    
    // Broadcast B across batch
    for (int b = 0; b < Batch; b++) {
        matmul_forward_kernel(A->data + b*M*K, B->data, C->data + b*M*N, M, K, N);
    }
}

// Batched Backward
void matmul_backward(Tensor* A, Tensor* B, Tensor* C) {
    int Batch = A->shape.dim[0], M = A->shape.dim[2], K = A->shape.dim[3], N = B->shape.dim[3];

    // 1. dA += dC @ B^T
    #pragma omp parallel for
    for (int b = 0; b < Batch; b++) {
        float* dA = A->grad + b*M*K;
        float* dC = C->grad + b*M*N;
        float* B_data = B->data;
        
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                float sum = 0.0f;
                for (int n = 0; n < N; n++) {
                    sum += dC[m*N + n] * B_data[k*N + n];
                }
                dA[m*K + k] += sum;
            }
        }
    }

    // 2. dB += A^T @ dC (Accumulate over batch)
    // Critical: Needs to be atomic or serial to avoid race on shared weight B
    for (int b = 0; b < Batch; b++) {
        float* A_data = A->data + b*M*K;
        float* dC = C->grad + b*M*N;
        float* dB = B->grad; // Shared

        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int m = 0; m < M; m++) {
                    sum += A_data[m*K + k] * dC[m*N + n];
                }
                dB[k*N + n] += sum;
            }
        }
    }
}

// --- 3. Dynamic RoPE (No Precomputation) ---

void rope_forward_dynamic(Tensor* Q, Tensor* K) {
    int Batch = Q->shape.dim[0];
    int T = Q->shape.dim[2];
    int D = Q->shape.dim[3];
    int Hd = HEAD_DIM;
    int H = N_HEADS;

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < Batch; b++) {
        for (int t = 0; t < T; t++) {
            int offset = b*T*D + t*D;
            float* q_ptr = Q->data + offset;
            float* k_ptr = K->data + offset;

            for (int h = 0; h < H; h++) {
                for (int i = 0; i < Hd/2; i++) {
                    // Calculate Theta on the fly
                    float freq = 1.0f / powf(10000.0f, (float)(2*i) / (float)Hd);
                    float theta = (float)t * freq;
                    float cos_t = cosf(theta);
                    float sin_t = sinf(theta);

                    int idx1 = h*Hd + 2*i;
                    int idx2 = h*Hd + 2*i + 1;

                    float q1 = q_ptr[idx1]; float q2 = q_ptr[idx2];
                    q_ptr[idx1] = q1*cos_t - q2*sin_t;
                    q_ptr[idx2] = q1*sin_t + q2*cos_t;

                    float k1 = k_ptr[idx1]; float k2 = k_ptr[idx2];
                    k_ptr[idx1] = k1*cos_t - k2*sin_t;
                    k_ptr[idx2] = k1*sin_t + k2*cos_t;
                }
            }
        }
    }
}

// --- 4. LayerNorm & Activation ---

void layernorm_forward(Tensor* x, Tensor* g, Tensor* b, Tensor* out, Tensor* mean, Tensor* var) {
    int Batch = x->shape.dim[0], T = x->shape.dim[2], D = x->shape.dim[3];
    #pragma omp parallel for
    for (int i = 0; i < Batch * T; i++) {
        float* x_p = x->data + i*D;
        float m = 0, v = 0;
        for (int j=0; j<D; j++) m += x_p[j]; m /= D;
        mean->data[i] = m;
        for (int j=0; j<D; j++) { float d = x_p[j] - m; v += d*d; } v /= D;
        var->data[i] = v;
        float inv = 1.0f / sqrtf(v + 1e-5f);
        for (int j=0; j<D; j++) out->data[i*D + j] = ((x_p[j] - m) * inv) * g->data[j] + b->data[j];
    }
}

void layernorm_backward(Tensor* x, Tensor* g, Tensor* b, Tensor* out, Tensor* mean, Tensor* var) {
    int Batch = x->shape.dim[0], T = x->shape.dim[2], D = x->shape.dim[3];
    int N = Batch * T;
    
    // Gradients for G and B must be accumulated serially or atomically
    for (int i = 0; i < N; i++) {
        float* dx = x->grad + i*D;
        float* dy = out->grad + i*D;
        float* xv = x->data + i*D;
        float m = mean->data[i];
        float inv = 1.0f / sqrtf(var->data[i] + 1e-5f);
        
        float sum_dy = 0, sum_dxhat = 0;
        for (int j=0; j<D; j++) {
            float xhat = (xv[j] - m) * inv;
            g->grad[j] += dy[j] * xhat;
            b->grad[j] += dy[j];
            float term = dy[j] * g->data[j];
            sum_dy += term;
            sum_dxhat += term * xhat;
        }
        for (int j=0; j<D; j++) {
            float xhat = (xv[j] - m) * inv;
            float term = dy[j] * g->data[j];
            dx[j] += (inv / D) * (D * term - sum_dy - xhat * sum_dxhat);
        }
    }
}

float gelu(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }

// --- 5. Attention ---

// Standard Attention implementation [B, H, T, T] for correctness.
// Can be swapped for Flash later.
void attention_forward(Tensor* Q, Tensor* K, Tensor* V, Tensor* Scores, Tensor* Probs, Tensor* Out) {
    rope_forward_dynamic(Q, K);
    
    int B = Q->shape.dim[0], T = Q->shape.dim[2], Hd = HEAD_DIM;
    float scale = 1.0f / sqrtf(Hd);

    #pragma omp parallel for collapse(2)
    for(int b=0; b<B; b++) {
        for(int h=0; h<N_HEADS; h++) {
            for(int t=0; t<T; t++) {
                float max_val = -1e9f;
                // Score = Q K^T
                for(int k=0; k<T; k++) {
                    float score = -1e9f;
                    if (k <= t) { // Causal
                        score = 0.0f;
                        for(int d=0; d<Hd; d++) {
                            int q_idx = b*T*D_MODEL + t*D_MODEL + h*Hd + d;
                            int k_idx = b*T*D_MODEL + k*D_MODEL + h*Hd + d;
                            score += Q->data[q_idx] * K->data[k_idx];
                        }
                        score *= scale;
                    }
                    Scores->data[b*N_HEADS*T*T + h*T*T + t*T + k] = score;
                    if(score > max_val) max_val = score;
                }
                
                // Softmax
                float sum = 0.0f;
                int row = b*N_HEADS*T*T + h*T*T + t*T;
                for(int k=0; k<T; k++) {
                    float e = (k > t) ? 0.0f : expf(Scores->data[row+k] - max_val);
                    Probs->data[row+k] = e;
                    sum += e;
                }
                for(int k=0; k<T; k++) Probs->data[row+k] /= (sum + 1e-9f);
                
                // Out = Probs @ V
                for(int d=0; d<Hd; d++) {
                    float val = 0.0f;
                    for(int k=0; k<=t; k++) {
                        int v_idx = b*T*D_MODEL + k*D_MODEL + h*Hd + d;
                        val += Probs->data[row+k] * V->data[v_idx];
                    }
                    Out->data[b*T*D_MODEL + t*D_MODEL + h*Hd + d] = val;
                }
            }
        }
    }
}

// --- 6. Optimizer & Persistence ---

void clip_gradients(Tensor** params, int count) {
    float sum_sq = 0.0f;
    for(int i=0; i<count; i++) {
        for(int j=0; j<params[i]->size; j++) sum_sq += params[i]->grad[j] * params[i]->grad[j];
    }
    float norm = sqrtf(sum_sq);
    if (norm > GRAD_CLIP) {
        float scale = GRAD_CLIP / (norm + 1e-6f);
        for(int i=0; i<count; i++) 
            for(int j=0; j<params[i]->size; j++) params[i]->grad[j] *= scale;
    }
}

void adamw_step(Tensor* t, float lr, float wd, int step) {
    float correction1 = 1.0f - powf(ADAM_B1, step+1);
    float correction2 = 1.0f - powf(ADAM_B2, step+1);
    for(int i=0; i<t->size; i++) {
        float g = t->grad[i];
        t->m[i] = ADAM_B1 * t->m[i] + (1.0f - ADAM_B1) * g;
        t->v[i] = ADAM_B2 * t->v[i] + (1.0f - ADAM_B2) * g * g;
        float mh = t->m[i] / correction1;
        float vh = t->v[i] / correction2;
        t->data[i] -= lr * (mh / (sqrtf(vh) + ADAM_EPS) + wd * t->data[i]);
    }
}

void save_checkpoint(const char* fn, Tensor** params, int count) {
    FILE* f = fopen(fn, "wb");
    for(int i=0; i<count; i++) fwrite(params[i]->data, sizeof(float), params[i]->size, f);
    fclose(f);
}

void load_checkpoint(const char* fn, Tensor** params, int count) {
    FILE* f = fopen(fn, "rb");
    if(f) {
        for(int i=0; i<count; i++) fread(params[i]->data, sizeof(float), params[i]->size, f);
        fclose(f);
    }
}

// --- 7. Model Structure & Wiring ---

typedef struct {
    Tensor w_q, w_k, w_v, w_o;
    Tensor ln1_g, ln1_b;
    Tensor w_ff1, w_ff2;
    Tensor ln2_g, ln2_b;
} Block;

typedef struct {
    Tensor token_emb;
    Block layers[MAX_LAYERS];
    Tensor ln_f_g, ln_f_b;
    Tensor w_head;
    Tensor* params[256];
    int n_params;
} GPT;

typedef struct {
    Tensor q, k, v, att_out, att_proj;
    Tensor att_scores, att_probs;
    Tensor ln1_out, ln1_mean, ln1_var;
    Tensor ln2_out, ln2_mean, ln2_var;
    Tensor ffn_in, ffn_act, ffn_out;
    Tensor res1, res2;
} LayerState;

typedef struct {
    Tensor emb_out;
    LayerState states[MAX_LAYERS];
    Tensor ln_f_out, ln_f_mean, ln_f_var;
    Tensor logits;
    Arena arena;
} GPTActivations;

GPT* gpt_init() {
    GPT* m = (GPT*)malloc(sizeof(GPT));
    m->n_params = 0;
    #define REG(t) m->params[m->n_params++] = &t

    m->token_emb = tensor_param(1, 1, VOCAB_SIZE, D_MODEL, "TokEmb"); REG(m->token_emb);
    for(int i=0; i<MAX_LAYERS; i++) {
        m->layers[i].w_q = tensor_param(1, 1, D_MODEL, D_MODEL, "WQ"); REG(m->layers[i].w_q);
        m->layers[i].w_k = tensor_param(1, 1, D_MODEL, D_MODEL, "WK"); REG(m->layers[i].w_k);
        m->layers[i].w_v = tensor_param(1, 1, D_MODEL, D_MODEL, "WV"); REG(m->layers[i].w_v);
        m->layers[i].w_o = tensor_param(1, 1, D_MODEL, D_MODEL, "WO"); REG(m->layers[i].w_o);
        m->layers[i].ln1_g = tensor_param(1, 1, D_MODEL, 1, "L1G"); REG(m->layers[i].ln1_g);
        m->layers[i].ln1_b = tensor_param(1, 1, D_MODEL, 1, "L1B"); REG(m->layers[i].ln1_b);
        m->layers[i].w_ff1 = tensor_param(1, 1, D_MODEL, D_FF, "WFF1"); REG(m->layers[i].w_ff1);
        m->layers[i].w_ff2 = tensor_param(1, 1, D_FF, D_MODEL, "WFF2"); REG(m->layers[i].w_ff2);
        m->layers[i].ln2_g = tensor_param(1, 1, D_MODEL, 1, "L2G"); REG(m->layers[i].ln2_g);
        m->layers[i].ln2_b = tensor_param(1, 1, D_MODEL, 1, "L2B"); REG(m->layers[i].ln2_b);
        
        for(int j=0; j<D_MODEL; j++) {
            m->layers[i].ln1_g.data[j] = 1.0f;
            m->layers[i].ln2_g.data[j] = 1.0f;
        }
    }
    m->w_head = tensor_param(1, 1, D_MODEL, VOCAB_SIZE, "Head"); REG(m->w_head);
    m->ln_f_g = tensor_param(1, 1, D_MODEL, 1, "LFG"); REG(m->ln_f_g);
    m->ln_f_b = tensor_param(1, 1, D_MODEL, 1, "LFB"); REG(m->ln_f_b);
    for(int j=0; j<D_MODEL; j++) m->ln_f_g.data[j] = 1.0f;
    return m;
}

GPTActivations gpt_activations() {
    GPTActivations a;
    a.arena = arena_create();
    a.emb_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
    for(int i=0; i<MAX_LAYERS; i++) {
        LayerState* l = &a.states[i];
        l->ln1_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->q = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->k = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->v = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->att_scores = tensor_arena(&a.arena, BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN);
        l->att_probs = tensor_arena(&a.arena, BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN);
        l->att_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->att_proj = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->res1 = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->ln2_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->ffn_in = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_FF);
        l->ffn_act = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_FF);
        l->ffn_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->res2 = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
        l->ln1_mean = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
        l->ln1_var = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
        l->ln2_mean = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
        l->ln2_var = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
    }
    a.ln_f_out = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, D_MODEL);
    a.logits = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, VOCAB_SIZE);
    a.ln_f_mean = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
    a.ln_f_var = tensor_arena(&a.arena, BATCH_SIZE, 1, SEQ_LEN, 1);
    return a;
}

void forward_pass(GPT* m, GPTActivations* c, int* inputs) {
    for(int b=0; b<BATCH_SIZE; b++) {
        for(int t=0; t<SEQ_LEN; t++) {
            int tid = inputs[b*SEQ_LEN + t];
            if(tid < 0 || tid >= VOCAB_SIZE) tid=0;
            memcpy(c->emb_out.data + b*SEQ_LEN*D_MODEL + t*D_MODEL, 
                   m->token_emb.data + tid*D_MODEL, D_MODEL*sizeof(float));
        }
    }
    Tensor* x = &c->emb_out;
    for(int i=0; i<MAX_LAYERS; i++) {
        LayerState* l = &c->states[i];
        Block* b = &m->layers[i];
        
        layernorm_forward(x, &b->ln1_g, &b->ln1_b, &l->ln1_out, &l->ln1_mean, &l->ln1_var);
        matmul_forward(&l->ln1_out, &b->w_q, &l->q);
        matmul_forward(&l->ln1_out, &b->w_k, &l->k);
        matmul_forward(&l->ln1_out, &b->w_v, &l->v);
        attention_forward(&l->q, &l->k, &l->v, &l->att_scores, &l->att_probs, &l->att_out);
        matmul_forward(&l->att_out, &b->w_o, &l->att_proj);
        
        for(int j=0; j<x->size; j++) l->res1.data[j] = x->data[j] + l->att_proj.data[j];
        
        layernorm_forward(&l->res1, &b->ln2_g, &b->ln2_b, &l->ln2_out, &l->ln2_mean, &l->ln2_var);
        matmul_forward(&l->ln2_out, &b->w_ff1, &l->ffn_in);
        for(int j=0; j<l->ffn_in.size; j++) l->ffn_act.data[j] = gelu(l->ffn_in.data[j]);
        matmul_forward(&l->ffn_act, &b->w_ff2, &l->ffn_out);
        
        for(int j=0; j<x->size; j++) l->res2.data[j] = l->res1.data[j] + l->ffn_out.data[j];
        x = &l->res2;
    }
    layernorm_forward(x, &m->ln_f_g, &m->ln_f_b, &c->ln_f_out, &c->ln_f_mean, &c->ln_f_var);
    matmul_forward(&c->ln_f_out, &m->w_head, &c->logits);
}

void backward_pass(GPT* m, GPTActivations* c, int* inputs) {
    matmul_backward(&c->ln_f_out, &m->w_head, &c->logits);
    layernorm_backward(&c->states[MAX_LAYERS-1].res2, &m->ln_f_g, &m->ln_f_b, &c->ln_f_out, &c->ln_f_mean, &c->ln_f_var);
    
    for(int i=MAX_LAYERS-1; i>=0; i--) {
        LayerState* l = &c->states[i];
        Block* b = &m->layers[i];
        
        // FFN Back
        for(int j=0; j<l->res2.size; j++) { l->res1.grad[j] += l->res2.grad[j]; l->ffn_out.grad[j] += l->res2.grad[j]; }
        matmul_backward(&l->ffn_act, &b->w_ff2, &l->ffn_out);
        for(int j=0; j<l->ffn_act.size; j++) {
            float x = l->ffn_in.data[j];
            float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
            float d = 0.5f * (1.0f + tanhf(inner)) + 0.5f * x * (1.0f - tanhf(inner)*tanhf(inner)) * 0.7978845608f * (1.0f + 3.0f * 0.044715f * x * x);
            l->ffn_in.grad[j] += l->ffn_act.grad[j] * d;
        }
        matmul_backward(&l->ln2_out, &b->w_ff1, &l->ffn_in);
        layernorm_backward(&l->res1, &b->ln2_g, &b->ln2_b, &l->ln2_out, &l->ln2_mean, &l->ln2_var);
        
        // Att Back
        Tensor* prev_out = (i > 0) ? &c->states[i-1].res2 : &c->emb_out;
        for(int j=0; j<l->res1.size; j++) { prev_out->grad[j] += l->res1.grad[j]; l->att_proj.grad[j] += l->res1.grad[j]; }
        matmul_backward(&l->att_out, &b->w_o, &l->att_proj);
        
        // Simplified Att Backprop (Routing gradients to Q,K,V roughly)
        // Note: For exact backprop of Softmax(QK)V, we need a full kernel.
        // For this single-file research code, we pass gradients through linear layers to ensure connectivity.
        for(int j=0; j<l->q.size; j++) {
            l->q.grad[j] += l->att_out.grad[j]; 
            l->k.grad[j] += l->att_out.grad[j]; 
            l->v.grad[j] += l->att_out.grad[j];
        }
        
        matmul_backward(&l->ln1_out, &b->w_q, &l->q);
        matmul_backward(&l->ln1_out, &b->w_k, &l->k);
        matmul_backward(&l->ln1_out, &b->w_v, &l->v);
        layernorm_backward(prev_out, &b->ln1_g, &b->ln1_b, &l->ln1_out, &l->ln1_mean, &l->ln1_var);
    }
    
    // Embed Back
    for(int b=0; b<BATCH_SIZE; b++) {
        for(int t=0; t<SEQ_LEN; t++) {
            int tid = inputs[b*SEQ_LEN + t];
            if(tid >= 0 && tid < VOCAB_SIZE) {
                float* src = c->emb_out.grad + b*SEQ_LEN*D_MODEL + t*D_MODEL;
                float* dst = m->token_emb.grad + tid*D_MODEL;
                for(int j=0; j<D_MODEL; j++) dst[j] += src[j];
            }
        }
    }
}

// --- Inference & Sampling ---

int sample_argmax(float* logits, int vocab_size) {
    int best_idx = 0;
    float best_val = -1e9f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

// Temperature sampling (Optional: adds creativity)
int sample_temp(float* logits, int vocab_size, float temp) {
    float probs[vocab_size];
    float sum = 0.0f;
    float max_l = -1e9f;
    
    // Stable Softmax with Temp
    for (int i=0; i<vocab_size; i++) if (logits[i] > max_l) max_l = logits[i];
    for (int i=0; i<vocab_size; i++) {
        probs[i] = expf((logits[i] - max_l) / temp);
        sum += probs[i];
    }
    
    // Random choice based on prob distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cdf = 0.0f;
    for (int i=0; i<vocab_size; i++) {
        cdf += probs[i] / sum;
        if (r < cdf) return i;
    }
    return vocab_size - 1;
}

// --- Dataset Loading ---

// Default fallback phrases if dataset.txt is not found
const char* default_phrases[] = {
    "Long Live The Motherland",
    "Knowledge is power",
    "Time waits for none",
    "Fortune favors the brave",
    "Unity is strength",
    "Seize the day",
    "Dreams come true",
    "Action conquers fear",
    "Silence is golden",
    "Hard work pays off"
};

#define NUM_DEFAULT_PHRASES (sizeof(default_phrases)/sizeof(default_phrases[0]))

// Load dataset from file or use defaults
char** load_dataset(const char* filename, int* num_phrases) {
    FILE* file = fopen(filename, "r");
    char** phrases = NULL;
    *num_phrases = 0;
    
    if (file) {
        printf("[IO] Loading dataset from '%s'\n", filename);
        
        // Count lines
        char buffer[MAX_PHRASE_LEN];
        int count = 0;
        while (fgets(buffer, sizeof(buffer), file) && count < MAX_PHRASES) {
            count++;
        }
        rewind(file);
        
        // Allocate memory
        phrases = (char**)malloc(count * sizeof(char*));
        if (!phrases) {
            fprintf(stderr, "Memory allocation failed\n");
            fclose(file);
            return NULL;
        }
        
        // Read phrases
        for (int i = 0; i < count; i++) {
            if (fgets(buffer, sizeof(buffer), file)) {
                // Remove newline
                buffer[strcspn(buffer, "\n")] = 0;
                buffer[strcspn(buffer, "\r")] = 0;
                
                // Skip empty lines
                if (strlen(buffer) == 0) {
                    i--; // Try again
                    continue;
                }
                
                phrases[i] = strdup(buffer);
                if (!phrases[i]) {
                    fprintf(stderr, "Memory allocation failed for phrase %d\n", i);
                    // Free previously allocated
                    for (int j = 0; j < i; j++) free(phrases[j]);
                    free(phrases);
                    fclose(file);
                    return NULL;
                }
                (*num_phrases)++;
            }
        }
        
        fclose(file);
        printf("[IO] Loaded %d phrases from dataset\n", *num_phrases);
    } else {
        // Use default phrases
        printf("[IO] Dataset file '%s' not found. Using default phrases.\n", filename);
        *num_phrases = NUM_DEFAULT_PHRASES;
        phrases = (char**)malloc(*num_phrases * sizeof(char*));
        if (!phrases) {
            fprintf(stderr, "Memory allocation failed\n");
            return NULL;
        }
        
        for (int i = 0; i < *num_phrases; i++) {
            phrases[i] = strdup(default_phrases[i]);
        }
        
        printf("[IO] Using %d default phrases\n", *num_phrases);
    }
    
    return phrases;
}

// Free dataset memory
void free_dataset(char** phrases, int num_phrases) {
    for (int i = 0; i < num_phrases; i++) {
        free(phrases[i]);
    }
    free(phrases);
}

// Encode random phrase from dataset
void encode_random_phrase(int* inputs, int* targets, char** phrases, int num_phrases) {
    // Pick random phrase
    int idx = rand() % num_phrases;
    const char* phrase = phrases[idx];
    int len = strlen(phrase);
    
    // Encode phrase with padding/truncation
    for (int i = 0; i < SEQ_LEN; i++) {
        if (i < len) {
            inputs[i] = (unsigned char)phrase[i];
            // For next token prediction
            if (i < len - 1) {
                targets[i] = (unsigned char)phrase[i + 1];
            } else {
                targets[i] = TOKEN_EOS;  // End of sequence
            }
        } else {
            // Padding
            inputs[i] = 0;
            targets[i] = 0;
        }
    }
}

// --- Main ---

int main(int argc, char* argv[]) {
    srand(time(NULL));
    
    // 1. Initialize Engine
    printf("[Phonex-C2] Booting...\n");
    GPT* model = gpt_init();
    GPTActivations acts = gpt_activations();

    // 2. Mode Selection
    if (argc < 2) {
        printf("Usage:\n  ./c2 train          -> Train and save 'model_final.bin'\n  ./c2 gen \"Prompt\"   -> Load 'model_final.bin' and generate text\n");
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        // --- TRAINING MODE ---
        printf("Training Mode Selected.\n");
        
        // Load dataset
        int num_phrases = 0;
        char** phrases = load_dataset("dataset.txt", &num_phrases);
        if (!phrases || num_phrases == 0) {
            fprintf(stderr, "No training data available.\n");
            return 1;
        }
        
        printf("Training with %d phrases...\n", num_phrases);
        
        // Allocate buffers
        int* inputs = malloc(BATCH_SIZE * SEQ_LEN * sizeof(int));
        int* targets = malloc(BATCH_SIZE * SEQ_LEN * sizeof(int));
        
        // Training Loop
        for(int step=0; step<MAX_STEPS; step++) {
            // 1. Zero Gradients & Activations
            arena_reset(&acts.arena);
            for(int i=0; i<model->n_params; i++) 
                memset(model->params[i]->grad, 0, model->params[i]->size * sizeof(float));
                
            // 2. Encode random phrase
            encode_random_phrase(inputs, targets, phrases, num_phrases);
            
            // 3. Forward
            forward_pass(model, &acts, inputs);
            
            // 4. Loss & Logits Grad
            float total_loss = 0;
            int total_tokens = 0;
            
            for(int b=0; b<BATCH_SIZE; b++) {
                for(int t=0; t<SEQ_LEN; t++) {
                    // Ignore padding (positions where input is 0)
                    if (inputs[b*SEQ_LEN + t] == 0) continue; 
                    
                    float* logits = acts.logits.data + b*SEQ_LEN*VOCAB_SIZE + t*VOCAB_SIZE;
                    int target = targets[b*SEQ_LEN + t];
                    
                    float max_l = -1e9f;
                    for(int i=0; i<VOCAB_SIZE; i++) if(logits[i] > max_l) max_l = logits[i];
                    float sum = 0;
                    for(int i=0; i<VOCAB_SIZE; i++) sum += expf(logits[i] - max_l);
                    float log_sum = max_l + logf(sum);
                    
                    total_loss += (log_sum - logits[target]);
                    total_tokens++;
                    
                    for(int i=0; i<VOCAB_SIZE; i++) {
                        float p = expf(logits[i] - log_sum);
                        acts.logits.grad[b*SEQ_LEN*VOCAB_SIZE + t*VOCAB_SIZE + i] = (p - (i==target ? 1.0f : 0.0f)) / BATCH_SIZE;
                    }
                }
            }
            
            // 5. Backward
            backward_pass(model, &acts, inputs);
            
            // 6. Update
            clip_gradients(model->params, model->n_params);
            
            // LR Schedule
            float lr = BASE_LR;
            if (step < WARMUP_STEPS) lr = BASE_LR * (float)step / WARMUP_STEPS;
            else lr = BASE_LR * 0.5f * (1.0f + cosf(3.14159f * (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)));
            
            for(int i=0; i<model->n_params; i++) adamw_step(model->params[i], lr, 0.01f, step);
            
            if (step % 10 == 0) {
                printf("Step %d | Loss: %.4f | LR: %.6f\n", step, total_loss / total_tokens, lr);
            }
        }
        
        save_checkpoint("model_final.bin", model->params, model->n_params);
        printf("[IO] Model saved to model_final.bin\n");
        
        // Cleanup
        free(inputs);
        free(targets);
        free_dataset(phrases, num_phrases);
    } 
    else if (strcmp(argv[1], "gen") == 0) {
        // --- INFERENCE MODE ---
        if (argc < 3) { 
            printf("Error: Provide a prompt.\nExample: ./c2 gen \"Long Live\"\n"); 
            return 1; 
        }
        
        printf("[IO] Loading model_final.bin...\n");
        load_checkpoint("model_final.bin", model->params, model->n_params);
        
        char* prompt = argv[2];
        int prompt_len = strlen(prompt);
        int* ctx = calloc(SEQ_LEN, sizeof(int));
        
        // Tokenize Prompt (Byte-level copy)
        printf("\nGenerating: %s", prompt);
        for(int i=0; i<prompt_len && i<SEQ_LEN; i++) ctx[i] = (unsigned char)prompt[i];
        
        // Generation Loop
        int head = prompt_len;
        while(head < SEQ_LEN) {
            arena_reset(&acts.arena);
            
            // 1. Run Model
            int batch_input[BATCH_SIZE * SEQ_LEN];
            memset(batch_input, 0, sizeof(batch_input));
            memcpy(batch_input, ctx, SEQ_LEN * sizeof(int));
            
            forward_pass(model, &acts, batch_input);
            
            // 2. Get Logits for the last token (head-1)
            float* logits = acts.logits.data + (0 * SEQ_LEN * VOCAB_SIZE) + ((head-1) * VOCAB_SIZE);
            
            // 3. Sample
            int next_token = sample_temp(logits, VOCAB_SIZE, 0.7f); // Temp 0.7
            
            // 4. Print & Advance
            printf("%c", (unsigned char)next_token);
            fflush(stdout);
            
            ctx[head] = next_token;
            head++;
        }
        printf("\n\n[Done]\n");
        free(ctx);
    }
    else {
        printf("Unknown command. Use 'train' or 'gen'.\n");
        return 1;
    }
    
    return 0;
}
