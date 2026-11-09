#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#include "arena.h"
#include "environment.h"
#include "model.h"
#include "training.h"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf(">>> Program started. If you can read this line, your console is working. <<<\n");
    fflush(stdout);

    mem_arena* arena = arena_create(MiB(64));
    if (!arena) {
        fprintf(stderr, "Failed to allocate memory arena!\n");
        return 1;
    }

    model_state* model = PUSH_STRUCT(arena, model_state);
    if (!model) {
        fprintf(stderr, "Failed to allocate model structure!\n");
        arena_destroy(arena);
        return 1;
    }

    create_actor_model(arena, model, BATCH);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);

    if (model->forward_graph.vars == NULL || model->cost_graph.vars == NULL) {
        fprintf(stderr, "Failed to build compute graph!\n");
        arena_destroy(arena);
        return 1;
    }

    SnakeENV* env = create_env(36);
    if (!env) {
        fprintf(stderr, "Failed to create environment!\n");
        arena_destroy(arena);
        return 1;
    }

    printf("Starting Training Loop...\n");
    train(model, env, arena);

    free(env);
    arena_destroy(arena);
    return 0;
}
