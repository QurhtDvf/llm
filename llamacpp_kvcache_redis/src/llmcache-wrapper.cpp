
#include "llama.h"
#include <cstring>

extern "C" {

void * llmcache_load_model(const char * path, int n_gpu_layers) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;
    return llama_model_load_from_file(path, mp);
}

void * llmcache_new_context(void * model, uint32_t n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    return llama_init_from_model((llama_model *)model, cp);
}

void llmcache_free_model(void * model) {
    llama_model_free((llama_model *)model);
}

void llmcache_free_context(void * ctx) {
    llama_free((llama_context *)ctx);
}

int llmcache_decode_tokens(
        void          * ctx,
        const int32_t * tokens,
        int32_t         n_tokens) {
    llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens), n_tokens);
    return llama_decode((llama_context *)ctx, batch);
}

int llmcache_tokenize(
        void          * model,
        const char    * text,
        int32_t         text_len,
        int32_t       * out_tokens,
        int32_t         max_tokens) {
    const llama_vocab * vocab = llama_model_get_vocab((llama_model *)model);
    return llama_tokenize(
        vocab, text, text_len, out_tokens, max_tokens, true, false);
}

void llmcache_kv_clear(void * ctx) {
    // llama_get_memory で llama_memory_t を取得してから clear
    llama_memory_t mem = llama_get_memory((llama_context *)ctx);
    if (mem) llama_memory_clear(mem, true);
}

} // extern "C"
