/* Inference for Llama-2 Transformer model in C++ + SYCL */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <chrono>

// #include <oneapi/mkl.hpp>
#include <fstream>

//#include <sycl/sycl.hpp>
//#include <dpct/device.hpp>

#include "mha.h"
#include "sycl_kernels.h"

//using namespace dpct;

//using dtype = sycl::half;
//#define MAX_SEQ_LEN 2048

// ----------------------------------------------------------------------------
// Transformer model

typedef struct transformer_config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct transformer_weights {
    // token embedding table
    sycl::half *token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    sycl::half *rms_att_weight; // (layer, dim) rmsnorm weights
    sycl::half *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    sycl::half *wq; // (layer, dim, n_heads * head_size)
    sycl::half *wk; // (layer, dim, n_kv_heads * head_size)
    sycl::half *wv; // (layer, dim, n_kv_heads * head_size)
    sycl::half *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    sycl::half *w1; // (layer, hidden_dim, dim)
    sycl::half *w2; // (layer, dim, hidden_dim)
    sycl::half *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    sycl::half *rms_final_weight; // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    sycl::half *freq_cis_real; // (seq_len, head_size/2)
    sycl::half *freq_cis_imag; // (seq_len, head_size/2)
    // (optional) classifier weights for the logits, on the last layer
    sycl::half *wcls;
} TransformerWeights;

typedef struct transformer_runstate {
    // current wave of activations
    sycl::half *x;   // activation at current time stamp (dim,)
    sycl::half *xb;  // same, but inside a residual branch (dim,)
    sycl::half *xb2; // an additional buffer just for convenience (dim,)
    sycl::half *hb;  // buffer for hidden dimension in the ffn (hidden_dim,)
    sycl::half *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    sycl::half *q;   // query (dim,)
    sycl::half *att; // buffer for scores/attention values (n_heads, seq_len)
    sycl::half *logits_gpu16; // output logits
    float* logits_gpu32; // logits in GPU memory converted to float
    float* logits; // logits copied CPU side
    // kv cache
    sycl::half *key_cache; // (layer, seq_len, dim)
    sycl::half *val_cache; // (layer, seq_len, dim)
} RunState;

typedef struct transformer_struct{
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    int shared_weights;
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
    sycl::queue& q_ct1 = get_default_queue();
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;

    // allocated on the GPU
    s->x            = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->xb           = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->xb2          = sycl::malloc_device<sycl::half>(p->dim,                               q_ct1);
    s->hb           = sycl::malloc_device<sycl::half>(p->hidden_dim,                        q_ct1);
    s->hb2          = sycl::malloc_device<sycl::half>(p->hidden_dim,                        q_ct1);
    s->q            = sycl::malloc_device<sycl::half>(p->n_heads * MAX_SEQ_LEN * head_size, q_ct1);
    s->att          = sycl::malloc_device<sycl::half>(p->n_heads  * p->seq_len,             q_ct1);
    s->logits_gpu16 = sycl::malloc_device<sycl::half>(p->vocab_size,                        q_ct1);
    s->logits_gpu32 = sycl::malloc_device<float>(p->vocab_size,                             q_ct1);
    s->key_cache    = sycl::malloc_device<sycl::half>(p->n_layers * p->seq_len * kv_dim,    q_ct1); // potentially huge allocs
    s->val_cache    = sycl::malloc_device<sycl::half>(p->n_layers * p->seq_len * kv_dim,    q_ct1);

    // allocated on the CPU
    s->logits = (float*)malloc(p->vocab_size * sizeof(float));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->att || !s->key_cache || !s->val_cache
        || !s->logits_gpu16 || !s->logits_gpu32 || !s->logits) {
        fprintf(stderr, "malloc runstate failed!\n");
        exit(EXIT_FAILURE);
    }

    // q_ct1.memset(s->att, 0, p->n_heads * p->seq_len * sizeof(sycl::half));
    // q_ct1.memset(s->val_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(sycl::half));
    // q_ct1.wait();
}

void free_run_state(RunState *s) {
    sycl::queue& q_ct1 = get_default_queue();

    sycl::free(s->x, q_ct1);
    sycl::free(s->xb, q_ct1);
    sycl::free(s->xb2, q_ct1);
    sycl::free(s->hb, q_ct1);
    sycl::free(s->hb2, q_ct1);
    sycl::free(s->q, q_ct1);
    sycl::free(s->att, q_ct1);
    sycl::free(s->logits_gpu16, q_ct1);
    sycl::free(s->logits_gpu32, q_ct1);
    sycl::free(s->key_cache, q_ct1);
    sycl::free(s->val_cache, q_ct1);
    free(s->logits);
}

void malloc_weights(TransformerWeights *w, Config *p, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    int head_size = p->dim / p->n_heads;
    printf("p->vocab_size: %d\n", p->vocab_size);
    printf("p->dim: %d\n", p->dim);
    printf("p->hidden_dim: %d\n", p->hidden_dim);
    printf("p->n_layers: %d\n", p->n_layers);
    printf("p->n_heads: %d\n", p->n_heads);
    printf("p->n_kv_heads: %d\n", p->n_kv_heads);
    printf("p->seq_len: %d\n", p->seq_len);
    printf("head_size: %d\n", head_size);
    printf("shared_weights: %d\n", shared_weights);

    w->token_embedding_table = sycl::malloc_device<sycl::half>(p->vocab_size * p->dim,                              q_ct1);

    w->rms_att_weight        = sycl::malloc_device<sycl::half>(p->n_layers * p->dim,                                q_ct1);
    w->wq                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_heads * head_size),     q_ct1);
    w->wk                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_kv_heads * head_size),  q_ct1);
    w->wv                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * (p->n_kv_heads * head_size),  q_ct1);
    w->wo                    = sycl::malloc_device<sycl::half>(p->n_layers * (p->n_heads * head_size) * p->dim,     q_ct1);

    w->rms_ffn_weight        = sycl::malloc_device<sycl::half>(p->n_layers * p->dim,                                q_ct1);
    w->w1                    = sycl::malloc_device<sycl::half>(p->n_layers * p->hidden_dim * p->dim,                q_ct1);
    w->w2                    = sycl::malloc_device<sycl::half>(p->n_layers * p->dim * p->hidden_dim,                q_ct1);
    w->w3                    = sycl::malloc_device<sycl::half>(p->n_layers * p->hidden_dim * p->dim,                q_ct1);

    w->rms_final_weight      = sycl::malloc_device<sycl::half>(p->dim,                                              q_ct1);

    w->freq_cis_real         = sycl::malloc_device<sycl::half>(p->seq_len * head_size / 2,                          q_ct1);
    w->freq_cis_imag         = sycl::malloc_device<sycl::half>(p->seq_len * head_size / 2,                          q_ct1);

    if (shared_weights) {
        w->wcls = w->token_embedding_table;
    } else {
        w->wcls              = sycl::malloc_device<sycl::half>(p->vocab_size * p->dim,                              q_ct1);
    }

    // ensure all mallocs went fine
    if (!w->token_embedding_table || !w->rms_att_weight || !w->rms_ffn_weight
        || !w->wq || !w->wk || !w->wv || !w->wo || !w->w1 || !w->w2 || !w->w3 ||
        !w->rms_final_weight || !w->freq_cis_real || !w->freq_cis_imag || !w->wcls) {
        fprintf(stderr, "malloc weights failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_weights(TransformerWeights *w, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    sycl::free(w->token_embedding_table, q_ct1);

    sycl::free(w->rms_att_weight, q_ct1);
    sycl::free(w->wq, q_ct1);
    sycl::free(w->wk, q_ct1);
    sycl::free(w->wv, q_ct1);
    sycl::free(w->wo, q_ct1);

    sycl::free(w->rms_ffn_weight, q_ct1);
    sycl::free(w->w1, q_ct1);
    sycl::free(w->w2, q_ct1);
    sycl::free(w->w3, q_ct1);

    sycl::free(w->rms_final_weight, q_ct1);

    sycl::free(w->freq_cis_real, q_ct1);
    sycl::free(w->freq_cis_imag, q_ct1);

    if (!shared_weights) {
        sycl::free(w->wcls, q_ct1);
    }
}

int load_weight(void* w, int elements, FILE* f, void* scratchCPU) {
    // read data into host memory
    int count = fread(scratchCPU, sizeof(dtype), elements, f);
    if (count != elements) return 1;
    // copy data to device memory
    get_default_queue().memcpy(w, scratchCPU, elements * sizeof(dtype)).wait();

    printf(".");
    fflush(stdout);
    return 0;
}

int load_checkpoint_weights(TransformerWeights *w, Config *p, FILE *f, int shared_weights) {
    sycl::queue& q_ct1 = get_default_queue();

    int head_size = p->dim / p->n_heads;
    size_t scratch_size = p->n_layers * std::max(p->dim, p->hidden_dim) * p->dim;
    scratch_size = std::max((size_t)p->vocab_size * p->dim, scratch_size);
    scratch_size *= sizeof(sycl::half);
    void* scratchCPU = malloc(scratch_size);

    printf("Loading weights\n");
     // populate each weight
    if (load_weight(w->token_embedding_table, p->vocab_size * p->dim,                               f, scratchCPU)) return 1;

    if (load_weight(w->rms_att_weight,        p->n_layers   * p->dim,                               f, scratchCPU)) return 1;
    if (load_weight(w->wq,                    p->n_layers   * p->dim * (p->n_heads    * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wk,                    p->n_layers   * p->dim * (p->n_kv_heads * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wv,                    p->n_layers   * p->dim * (p->n_kv_heads * head_size), f, scratchCPU)) return 1;
    if (load_weight(w->wo,                    p->n_layers   * (p->n_heads * head_size) * p->dim,    f, scratchCPU)) return 1;

    if (load_weight(w->rms_ffn_weight,        p->n_layers   * p->dim,                               f, scratchCPU)) return 1;
    if (load_weight(w->w1,                    p->n_layers   * p->dim * p->hidden_dim,               f, scratchCPU)) return 1;
    if (load_weight(w->w2,                    p->n_layers   * p->hidden_dim * p->dim,               f, scratchCPU)) return 1;
    if (load_weight(w->w3,                    p->n_layers   * p->dim * p->hidden_dim,               f, scratchCPU)) return 1;

    if (load_weight(w->rms_final_weight,      p->dim,                                               f, scratchCPU)) return 1;

    if (load_weight(w->freq_cis_real,         p->seq_len    * head_size / 2,                        f, scratchCPU)) return 1;
    if (load_weight(w->freq_cis_imag,         p->seq_len    * head_size / 2,                        f, scratchCPU)) return 1;

    if (!shared_weights) {
        if (load_weight(w->wcls,              p->vocab_size * p->dim,                               f, scratchCPU)) return 1;
    }

    printf("\ndone\n");
    free(scratchCPU);
    return 0;
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    // open checkpoint
    FILE *file = fopen(checkpoint_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
    // int magic, version;
    // if (fread(&magic,   sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    // if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    t->shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    // fseek(file, 256, SEEK_SET);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    // allocate the Weights
    malloc_weights(&t->weights, &t->config, t->shared_weights);
    // read in the Config and the Weights from the checkpoint
    if (load_checkpoint_weights(&t->weights, &t->config, file, t->shared_weights)) { fprintf(stderr, "Couldn't load weights\n"); exit(EXIT_FAILURE); }
    fclose(file);
}

void free_transformer(Transformer* t) {
    // free the RunState buffers
    free_run_state(&t->state);
    // free the transformer weights
    free_weights(&t->weights, t->shared_weights);
}

void debug_print(sycl::half* device_data, int num_points = 128) {
    sycl::half* host_data = (sycl::half*)malloc(num_points * sizeof(sycl::half));
    get_default_queue().memcpy(host_data, device_data, num_points * sizeof(sycl::half)).wait();
    for (int i = 0; i < num_points; i++) {
        printf("%f ", (float)host_data[i]);
    }
    printf("\n");
    free(host_data);
}

void forward(Transformer* transformer, int token, int pos) {

    sycl::queue& q_ct1 = get_default_queue();

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    sycl::half *x = s->x;

    // copy the token embedding into x
    sycl::half *content_row = &(w->token_embedding_table[token * dim]);
    q_ct1.memcpy(x, content_row, dim * sizeof(sycl::half));

    // forward all the layers
    for (int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // we directly store (key, value) at this time step (pos) to our kv cache
        int loff = l * p->n_kv_heads * p->seq_len * head_size; // kv cache layer offset for convenience
        sycl::half *qrow = s->q + pos * head_size;
        sycl::half *krow = s->key_cache + loff + pos * head_size;
        sycl::half *vrow = s->val_cache + loff + pos * head_size;

        // qkv matmuls for this position
        // matmul(qrow, s->xb, w->wq + l*dim*   dim, dim,    dim);
        // matmul(krow, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        // matmul(vrow, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);
        matmul_qkv(qrow, krow, vrow, s->xb, w->wq + l*dim*dim, w->wk + l*dim*kv_dim, w->wv + l*dim*kv_dim, dim, head_size, p->n_heads);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        // also save the output (key, value) at this time step (pos) to our kv cache
        RoPERotation(qrow, krow, pos, p->n_heads, p->n_kv_heads, head_size);

        #ifdef _ENABLE_FLASH_ATTENTION
        //gpu::xetla::fmha_forward_index_kernel();
        gpu::xetla::fmha_forward_op(q_ct1, qrow, s->key_cache + loff, s->val_cache + loff, s->xb, 1, p->n_heads, head_size, 1, pos+1);
        #else
        // apply MHA using the query and the key-value cache
        MultiHeadAttention(s->xb, qrow, s->key_cache + loff, s->val_cache + loff, s->att, p->n_heads, head_size, pos+1);
        #endif

        // final matmul to get the output of the attention
        // matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        // accum(x, s->xb2, dim);
        matmul_mad(x, s->xb, w->wo + l*dim*dim, dim, dim);

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        // matmul(s->hb,  s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        // matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
        matmul_2X(s->hb, s->hb2, s->xb, w->w1 + l*dim*hidden_dim, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // apply F.silu activation on hb and multiply it with hb2
        siluElementwiseMul(s->hb, s->hb2, hidden_dim);

        // final matmul to get the output of the ffn
        // matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        // accum(x, s->xb, dim);
        matmul_mad(x, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    // matmul(s->logits_gpu16, x, w->wcls, p->dim, p->vocab_size);
    matmul_f32(s->logits_gpu32, x, w->wcls, p->dim, p->vocab_size);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct transformer_tokenindex {
    char *str;
    int id;
} TokenIndex;

typedef struct transformer_tokenizer {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*) malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct transformer_probindex{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct transformer_sampler{
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;


int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex*) malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, RunState* state) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(state->logits_gpu32, sampler->vocab_size);
    } else {
        // // apply the temperature to the logits
        // float inv_temperature = 1.0f / sampler->temperature;
        // scalar_mul32_kernel <<< divUp(sampler->vocab_size, 256), 256 >>> (state->logits_gpu32, inv_temperature, sampler->vocab_size);
        // // apply softmax to the logits to get the probabilities for next token
        // softmax32_kernel <<< 1, 1024 >>> (state->logits_gpu32, sampler->vocab_size);
        // // copy the logits from GPU to the CPU
        // cudaMemcpy(state->logits, state->logits_gpu32, sampler->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(state->logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(state->logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

struct struct_color
{
    const std::string GREEN = "\033[92m";
    const std::string BLUE  = "\033[94m";
    const std::string END   = "\033[0m";
} colors;

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    std::chrono::steady_clock::time_point time_start0;
    std::chrono::steady_clock::time_point time_start1;
    std::chrono::steady_clock::time_point time_start2;
    std::chrono::steady_clock::time_point time_start3;
    std::chrono::steady_clock::time_point time_stop;

    // start the main loop
    // long start0 = 0;  // used to time our code, only initialized after first iteration
    // long start1 = 0;  // used to time our code, only initialized after first iteration
    // long start2 = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence

    // start0 = time_in_ms();
    time_start0 = std::chrono::steady_clock::now();

    while (pos < steps) {

        // forward the transformer to get logits for the next token
        forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, &transformer->state);
        }
        pos++;

        //printf("#### generate token=%d, next=%d\n", token, next);

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        if (pos < num_prompt_tokens) {
            std::cout << colors.BLUE << piece << colors.END << std::flush;
        } else {
            std::cout << colors.GREEN << piece << colors.END << std::flush;
        }
        token = next;

        // init the timer here because the first iteration can be slower
        // if (start2 == 0) { start2 = time_in_ms(); }
        if (pos == num_prompt_tokens) {
            // start1 = time_in_ms();
            time_start1 = std::chrono::steady_clock::now();
        }
        if (pos == num_prompt_tokens + 1) {
            // setenv("PTI_ENABLE_COLLECTION", "1", 1);
            // start2 = time_in_ms();
            time_start2 = std::chrono::steady_clock::now();
        }
        if (pos == num_prompt_tokens + 2) {
            time_start3 = std::chrono::steady_clock::now();
        }
    }
    printf("\n");
    // unsetenv("PTI_ENABLE_COLLECTION");
    // report achieved tok/s (pos - num_prompt_tokens because the timer starts after first iteration)
    if (pos > num_prompt_tokens) {
        // long end = time_in_ms();
        time_stop = std::chrono::steady_clock::now();
        printf("input tokens: %d\n", num_prompt_tokens);
        printf("new tokens: %d\n", pos - num_prompt_tokens);

        // printf("achieved 1st token latency: %f ms\n", (double)(start2 - start1));
        // printf("achieved 1st token latency (incl input): %f ms\n", (double)(start2 - start0) / (num_prompt_tokens + 1));

        // printf("achieved tok/s: %f\n", (pos - num_prompt_tokens) / (double)(end - start2) * 1000);
        // printf("achieved next token latency: %f ms\n", (double)(end - start2) / (pos - num_prompt_tokens - 1));


        printf("achieved 1st token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_start2 - time_start1).count());
        // printf("achieved 1st token latency (incl input): %f ms\n", std::chrono::duration<double, std::milli>(time_start2 - time_start0).count() / (num_prompt_tokens + 1));

        printf("achieved 2nd token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_start3 - time_start2).count());
        // printf("achieved next token latency: %f ms\n", std::chrono::duration<double, std::milli>(time_stop - time_start2).count() / (pos - num_prompt_tokens - 1));
    }

    free(prompt_tokens);
}

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
        // generate(&transformer, &tokenizer, &sampler, prompt, steps);
        // generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        // chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
        fprintf(stderr, "chat mode not implemented\n");
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
