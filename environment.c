#include "environment.h"

#include "random.h"

#include <math.h>
#include <stdlib.h>

SnakeENV* create_env(u32 grid_size) {
    SnakeENV* env = malloc(sizeof(*env));
    if (!env) return NULL;

    u32 side = (u32)sqrtf((f32)grid_size);
    if (side * side != grid_size) { free(env); return NULL; }

    env->rows = side; env->cols = side; env->grid_size = grid_size;
    return env;
}

State get_random_food_loc(u32 cols, u32 rows) {
    return (State) { .x = (i32)randn(cols), .y = (i32)randn(rows) };
}

b32 state_out_of_bounds(State state, u32 cols, u32 rows) {
    return state.x < 0 || state.x >= (i32)cols || state.y < 0 || state.y >= (i32)rows;
}

void reset_batch_env(BatchEnv* batch_env, const SnakeENV* env) {
    for (u32 i = 0; i < BATCH; i++) {
        batch_env->snake[i] = (State){ .x = (i32)(env->cols / 2), .y = (i32)(env->rows / 2) };
        do {
            batch_env->food[i] = get_random_food_loc(env->cols, env->rows);
        } while (batch_env->food[i].x == batch_env->snake[i].x && batch_env->food[i].y == batch_env->snake[i].y);
        batch_env->pov[i] = RIGHT;
        batch_env->active[i] = true;
        batch_env->foods_eaten[i] = 0;
    }
}

f32 step_batch_env(BatchEnv* batch_env, u32 index, const SnakeENV* env, ACTION action) {
    if (action == NONE) action = batch_env->pov[index];
    if (action == LEFT) batch_env->snake[index].x -= 1;
    else if (action == RIGHT) batch_env->snake[index].x += 1;
    else if (action == UP) batch_env->snake[index].y += 1;
    else if (action == DOWN) batch_env->snake[index].y -= 1;
    batch_env->pov[index] = action;

    f32 reward = 0.0f;
    if (batch_env->snake[index].x == batch_env->food[index].x && batch_env->snake[index].y == batch_env->food[index].y) {
        reward += 20.0f;
        batch_env->foods_eaten[index]++;
        State new_food;
        do {
            new_food = get_random_food_loc(env->cols, env->rows);
        } while (new_food.x == batch_env->snake[index].x && new_food.y == batch_env->snake[index].y);
        batch_env->food[index] = new_food;
    }
    if (state_out_of_bounds(batch_env->snake[index], env->cols, env->rows)) {
        reward -= 10.0f;
    }
    return reward;
}

void build_state_vector_col(matrix* input, u32 col, State state, State food_state, ACTION pov, u32 cols, u32 grid_size) {
    u32 snake_i = (u32)state.y * cols + (u32)state.x;
    u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;
    input->data[col + (u64)snake_i * input->cols] = 1.0f;
    input->data[col + (u64)(grid_size + food_i) * input->cols] = 1.0f;
    input->data[col + (u64)(2 * grid_size + (u32)pov) * input->cols] = 1.0f;
}

ACTION sample_action_col(const matrix* probs, u32 col) {
    f32 r = prng_randf();
    f32 cdf = 0.0f;
    for (u32 i = 0; i < probs->rows; i++) {
        cdf += probs->data[col + (u64)i * probs->cols];
        if (r <= cdf) return (ACTION)i;
    }
    return (ACTION)(probs->rows - 1);
}