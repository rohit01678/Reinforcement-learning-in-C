#include "random.h"

static u64 prng_state = 0x853c42e635f12e17ULL;

static u64 prng_next(void) {
    u64 x = prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    prng_state = x;
    return x;
}

f32 prng_randf(void) {
    return (f32)(prng_next() & 0xFFFFFF) / (f32)0x1000000;
}

u32 randn(u32 max_val) {
    if (max_val == 0) return 0;
    return (u32)(prng_next() % max_val);
}