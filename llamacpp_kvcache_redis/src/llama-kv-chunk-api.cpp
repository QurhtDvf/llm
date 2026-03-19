#include "llama-kv-cache.h"
#include "llama-context.h"
#include "llama-io.h"
#include <cstring>
#include <vector>

struct kv_chunk_writer : public llama_io_write_i {
    std::vector<uint8_t> buf;
    void write(const void * src, size_t size) override {
        const auto * p = static_cast<const uint8_t *>(src);
        buf.insert(buf.end(), p, p + size);
    }
    void write_tensor(const ggml_tensor * t, size_t offset, size_t size) override {
        size_t prev = buf.size();
        buf.resize(prev + size);
        ggml_backend_tensor_get(t, buf.data() + prev, offset, size);
    }
    size_t n_bytes() override { return buf.size(); }
};

struct kv_chunk_reader : public llama_io_read_i {
    const uint8_t * data; size_t total; size_t pos = 0;
    kv_chunk_reader(const uint8_t * d, size_t s) : data(d), total(s) {}
    const uint8_t * read(size_t n) override {
        GGML_ASSERT(pos + n <= total);
        const uint8_t * p = data + pos; pos += n; return p;
    }
    void read_to(void * dst, size_t n) override {
        GGML_ASSERT(pos + n <= total);
        memcpy(dst, data + pos, n); pos += n;
    }
    size_t n_bytes() override { return pos; }
};

extern "C" {

bool llmcache_kv_write_chunk(
        llama_context * ctx, void * dst, size_t * dst_size,
        uint32_t token_start, uint32_t token_count) {
    try {
        kv_chunk_writer writer;
        auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
        if (!kv) return false;
        kv->state_write_chunk(writer, token_start, token_count);
        if (writer.buf.size() > *dst_size) {
            *dst_size = writer.buf.size(); return false;
        }
        memcpy(dst, writer.buf.data(), writer.buf.size());
        *dst_size = writer.buf.size();
        return true;
    } catch (...) { return false; }
}

bool llmcache_kv_read_chunk(
        llama_context * ctx, const void * src, size_t src_size,
        uint32_t token_start, uint32_t token_count) {
    try {
        kv_chunk_reader reader(static_cast<const uint8_t *>(src), src_size);
        auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
        if (!kv) return false;
        return kv->state_read_chunk(reader, token_start, token_count);
    } catch (...) { return false; }
}

size_t llmcache_kv_chunk_size(
        llama_context * ctx, uint32_t token_count) {
    auto * kv = dynamic_cast<llama_kv_cache *>(ctx->get_memory());
    if (!kv) return 0;
    return kv->get_chunk_size_bytes(token_count);
}

} // extern "C"
