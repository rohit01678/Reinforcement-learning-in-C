#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "common.h"

typedef enum { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3, NONE = 4 } ACTION;
typedef struct { i32 x, y; } State;

typedef struct {
    u32 rows, cols, grid_size;
} SnakeENV;

typedef struct {
    State snake[BATCH];
    State food[BATCH];
    ACTION pov[BATCH];
    b32 active[BATCH];
    u32 foods_eaten[BATCH];
} BatchEnv;

SnakeENV* create_env(u32 grid_size);
State get_random_food_loc(u32 cols, u32 rows);
b32 state_out_of_bounds(State state, u32 cols, u32 rows);
void reset_batch_env(BatchEnv* batch_env, const SnakeENV* env);
f32 step_batch_env(BatchEnv* batch_env, u32 index, const SnakeENV* env, ACTION action);
void build_state_vector_col(matrix* input, u32 col, State state, State food_state, ACTION pov, u32 cols, u32 grid_size);
ACTION sample_action_col(const matrix* probs, u32 col);

#endif