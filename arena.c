#include "arena.h"

#include <stdlib.h>
#include <string.h>

mem_arena* arena_create(u64 capacity) {
    mem_arena* arena = malloc(sizeof(mem_arena));
    if (arena == NULL) return NULL;

    arena->raw = malloc(capacity + ARENA_ALIGN);
    if (arena->raw == NULL) {
        free(arena);
        return NULL;
    }

    arena->memory = (u8*)ALIGN_UP_POW2((uintptr_t)arena->raw, (uintptr_t)ARENA_ALIGN);
    arena->capacity = capacity;
    arena->pos = 0;
    return arena;
}

void arena_destroy(mem_arena* arena) {
    if (arena == NULL) return;
    free(arena->raw);
    free(arena);
}

void* arena_push(mem_arena* arena, u64 size, b32 zero) {
    if (arena == NULL) return NULL;

    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
    if (pos_aligned > arena->capacity || size > arena->capacity - pos_aligned) {
        return NULL;
    }

    u64 new_pos = pos_aligned + size;
    u8* out = arena->memory + pos_aligned;
    arena->pos = new_pos;

    if (zero) {
        memset(out, 0, size);
    }

    return out;
}