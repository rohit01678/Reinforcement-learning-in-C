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
        batch_env->body_len[i] = 3;
        batch_env->body[i][0] = batch_env->snake[i];
        batch_env->body[i][1] = (State){ .x = batch_env->snake[i].x - 1, .y = batch_env->snake[i].y };
        batch_env->body[i][2] = (State){ .x = batch_env->snake[i].x - 2, .y = batch_env->snake[i].y };
        do {
            batch_env->food[i] = get_random_food_loc(env->cols, env->rows);
            b32 occupied = false;
            for (u32 segment = 0; segment < batch_env->body_len[i]; segment++) {
                if (batch_env->body[i][segment].x == batch_env->food[i].x && batch_env->body[i][segment].y == batch_env->food[i].y) occupied = true;
            }
            if (!occupied) break;
        } while (true);
        batch_env->pov[i] = RIGHT;
        batch_env->active[i] = true;
        batch_env->foods_eaten[i] = 0;
    }
}

f32 step_batch_env(BatchEnv* batch_env, u32 index, const SnakeENV* env, ACTION action) {
    ACTION current = batch_env->pov[index];
    if (action == NONE ||
        (current == LEFT && action == RIGHT) || (current == RIGHT && action == LEFT) ||
        (current == UP && action == DOWN) || (current == DOWN && action == UP)) action = current;

    State next = batch_env->snake[index];
    if (action == LEFT) next.x -= 1;
    else if (action == RIGHT) next.x += 1;
    else if (action == UP) next.y += 1;
    else if (action == DOWN) next.y -= 1;
    batch_env->pov[index] = action;

    State food = batch_env->food[index];
    i32 old_distance = abs(food.x - batch_env->snake[index].x) + abs(food.y - batch_env->snake[index].y);
    i32 new_distance = abs(food.x - next.x) + abs(food.y - next.y);
    f32 reward = 0.1f * (f32)(old_distance - new_distance) - 0.01f;

    if (state_out_of_bounds(next, env->cols, env->rows)) {
        batch_env->snake[index] = next;
        batch_env->active[index] = false;
        return reward - 10.0f;
    }

    b32 ate_food = next.x == batch_env->food[index].x && next.y == batch_env->food[index].y;
    u32 collision_len = batch_env->body_len[index] - (ate_food ? 0 : 1);
    for (u32 segment = 0; segment < collision_len; segment++) {
        if (batch_env->body[index][segment].x == next.x && batch_env->body[index][segment].y == next.y) {
            batch_env->snake[index] = next;
            batch_env->active[index] = false;
            return reward - 10.0f;
        }
    }

    if (ate_food) {
        reward += 20.0f;
        batch_env->foods_eaten[index]++;
        if (batch_env->body_len[index] == env->grid_size) {
            batch_env->active[index] = false;
            return reward;
        }
        batch_env->body_len[index]++;
    }

    for (u32 segment = batch_env->body_len[index] - 1; segment > 0; segment--) {
        batch_env->body[index][segment] = batch_env->body[index][segment - 1];
    }
    batch_env->body[index][0] = next;
    batch_env->snake[index] = next;

    if (ate_food) {
        do {
            batch_env->food[index] = get_random_food_loc(env->cols, env->rows);
            b32 occupied = false;
            for (u32 segment = 0; segment < batch_env->body_len[index]; segment++) {
                if (batch_env->body[index][segment].x == batch_env->food[index].x && batch_env->body[index][segment].y == batch_env->food[index].y) occupied = true;
            }
            if (!occupied) break;
        } while (true);
    }
    return reward;
}

void build_state_vector_col(matrix* input, u32 col, State state, State food_state, const State* body, u32 body_len, ACTION pov, u32 cols, u32 grid_size) {
    for (u32 segment = 0; segment < body_len; segment++) {
        u32 body_i = (u32)body[segment].y * cols + (u32)body[segment].x;
        input->data[col + (u64)body_i * input->cols] = 1.0f;
    }
    input->data[col + (u64)grid_size * input->cols] = (f32)state.x / (f32)(cols - 1);
    input->data[col + (u64)(grid_size + 1) * input->cols] = (f32)state.y / (f32)(grid_size / cols - 1);
    input->data[col + (u64)(grid_size + 2) * input->cols] = (f32)food_state.x / (f32)(cols - 1);
    input->data[col + (u64)(grid_size + 3) * input->cols] = (f32)food_state.y / (f32)(grid_size / cols - 1);
    input->data[col + (u64)(grid_size + 4 + (u32)pov) * input->cols] = 1.0f;
    input->data[col + (u64)(grid_size + 8) * input->cols] = (f32)body[body_len - 1].x / (f32)(cols - 1);
    input->data[col + (u64)(grid_size + 9) * input->cols] = (f32)body[body_len - 1].y / (f32)(grid_size / cols - 1);
}

static b32 action_is_valid(ACTION action, ACTION pov) {
    return !((pov == LEFT && action == RIGHT) || (pov == RIGHT && action == LEFT) ||
             (pov == UP && action == DOWN) || (pov == DOWN && action == UP));
}

static b32 same_state(State a, State b) {
    return a.x == b.x && a.y == b.y;
}

static b32 action_is_safe(ACTION action, ACTION pov, State head, State food, State tail, const State* body, u32 body_len, u32 cols, u32 rows) {
    if (!action_is_valid(action, pov)) return false;
    State next = head;
    if (action == LEFT) next.x--;
    else if (action == RIGHT) next.x++;
    else if (action == UP) next.y++;
    else if (action == DOWN) next.y--;
    if (state_out_of_bounds(next, cols, rows)) return false;
    b32 growing = same_state(next, food);
    for (u32 segment = 0; segment < body_len; segment++) {
        if (!growing && same_state(body[segment], tail)) continue;
        if (body[segment].x == next.x && body[segment].y == next.y) return false;
    }
    return true;
}

static u32 reachable_metrics(State start, State food, State tail, const State* body, u32 body_len, b32 growing, u32 cols, u32 rows, u32* food_distance) {
    b32 blocked[GRID_SIZE] = { 0 };
    b32 visited[GRID_SIZE] = { 0 };
    u32 distances[GRID_SIZE];
    u32 queue[GRID_SIZE];
    for (u32 cell = 0; cell < GRID_SIZE; cell++) distances[cell] = GRID_SIZE + 1;

    for (u32 segment = 0; segment < body_len; segment++) {
        if (!growing && same_state(body[segment], tail)) continue;
        u32 cell = (u32)body[segment].y * cols + (u32)body[segment].x;
        blocked[cell] = true;
    }

    u32 start_cell = (u32)start.y * cols + (u32)start.x;
    u32 food_cell = (u32)food.y * cols + (u32)food.x;
    u32 read_index = 0, write_index = 0, area = 1;
    queue[write_index++] = start_cell;
    visited[start_cell] = true;
    distances[start_cell] = 0;

    static const i32 dx[4] = { -1, 1, 0, 0 };
    static const i32 dy[4] = { 0, 0, 1, -1 };
    while (read_index < write_index) {
        u32 cell = queue[read_index++];
        i32 x = (i32)(cell % cols);
        i32 y = (i32)(cell / cols);
        for (u32 direction = 0; direction < 4; direction++) {
            State next = { .x = x + dx[direction], .y = y + dy[direction] };
            if (state_out_of_bounds(next, cols, rows)) continue;
            u32 next_cell = (u32)next.y * cols + (u32)next.x;
            if (blocked[next_cell] || visited[next_cell]) continue;
            visited[next_cell] = true;
            distances[next_cell] = distances[cell] + 1;
            queue[write_index++] = next_cell;
            area++;
        }
    }

    *food_distance = distances[food_cell];
    return area;
}

ACTION sample_action_col(const matrix* probs, u32 col, ACTION pov, State head, State food, State tail, const State* body, u32 body_len, u32 cols, u32 rows) {
    f32 total = 0.0f;
    for (u32 i = 0; i < 4; i++) {
        if (action_is_safe((ACTION)i, pov, head, food, tail, body, body_len, cols, rows)) total += probs->data[col + (u64)i * probs->cols];
    }
    if (total <= 0.0f) return pov;
    f32 r = prng_randf() * total;
    f32 cdf = 0.0f;
    ACTION last_valid = pov;
    for (u32 i = 0; i < 4; i++) {
        if (!action_is_safe((ACTION)i, pov, head, food, tail, body, body_len, cols, rows)) continue;
        last_valid = (ACTION)i;
        cdf += probs->data[col + (u64)i * probs->cols];
        if (r <= cdf) return (ACTION)i;
    }
    return last_valid;
}

ACTION best_action_col(const matrix* probs, u32 col, ACTION pov, State head, State food, State tail, const State* body, u32 body_len, u32 cols, u32 rows) {
    ACTION best = pov;
    f32 best_score = -INFINITY;
    for (u32 i = 0; i < 4; i++) {
        if (!action_is_safe((ACTION)i, pov, head, food, tail, body, body_len, cols, rows)) continue;
        f32 probability = probs->data[col + (u64)i * probs->cols];
        State next = head;
        if (i == LEFT) next.x--;
        else if (i == RIGHT) next.x++;
        else if (i == UP) next.y++;
        else if (i == DOWN) next.y--;
        u32 food_distance = GRID_SIZE + 1;
        b32 growing = next.x == food.x && next.y == food.y;
        u32 reachable = reachable_metrics(next, food, tail, body, body_len, growing, cols, rows, &food_distance);
        f32 space_score = 1.75f * logf((f32)reachable + 1.0f);
        f32 path_score = food_distance > GRID_SIZE ? -20.0f : -0.65f * (f32)food_distance;
        f32 score = probability + space_score + path_score;
        if (score > best_score) {
            best_score = score;
            best = (ACTION)i;
        }
    }
    return best;
}