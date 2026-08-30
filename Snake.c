#define _CRT_SECURE_NO_WARNINGS
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// BASE TYPES & DEFINITIONS
// ============================================================================
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef uint8_t b8;
typedef uint32_t b32;

typedef float f32;
typedef double f64;

#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define ALIGN_UP_POW2(x, align) (((x) + (align) - 1) & ~((align) - 1))

// ============================================================================
// MEMORY ARENA
// ============================================================================
#define ARENA_ALIGN (sizeof(void*))

typedef struct {
    u8* memory;
    u64 capacity;
    u64 pos;
} mem_arena;

mem_arena* arena_create(u64 capacity) {
    mem_arena* arena = malloc(sizeof(mem_arena));
    if (arena == NULL) return NULL;

    arena->memory = malloc(capacity);
    if (arena->memory == NULL) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->pos = 0;
    return arena;
}

void arena_destroy(mem_arena* arena) {
    if (arena == NULL) return;
    free(arena->memory);
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

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), true)
#define PUSH_ARRAY_UNINIT(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n), false)

// ============================================================================
// PRNG (PSEUDO-RANDOM NUMBER GENERATOR)
// ============================================================================
static u64 prng_state = 0x853c42e635f12e17ULL;

static inline u64 prng_next(void) {
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

// ============================================================================
// MATRIX & LINEAR ALGEBRA
// ============================================================================
typedef struct {
    u32 rows, cols;
    f32* data;
} matrix;

matrix* create_matrix(mem_arena* arena, u32 rows, u32 cols) {
    matrix* mat = PUSH_STRUCT(arena, matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, (u64)rows * cols);
    return mat;
}

void clear(matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void fill(matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) mat->data[i] = x;
}

void fill_rand(matrix* mat, f32 lower, f32 upper) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) {
        mat->data[i] = prng_randf() * (upper - lower) + lower;
    }
}

void scale(matrix* mat, f32 s) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) mat->data[i] *= s;
}

b32 add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows || out->cols != a->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = a->data[i] + b->data[i];
    return true;
}

b32 sub(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols || out->rows != a->rows || out->cols != a->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = a->data[i] - b->data[i];
    return true;
}

void _mat_mul_nn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            f32 sum = 0.0f;
            for (u64 k = 0; k < a->cols; k++) {
                sum += a->data[k + i * a->cols] * b->data[j + k * b->cols];
            }
            out->data[j + i * out->cols] += sum;
        }
    }
}

void _mat_mul_nt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            f32 sum = 0.0f;
            for (u64 k = 0; k < a->cols; k++) {
                sum += a->data[k + i * a->cols] * b->data[k + j * b->cols];
            }
            out->data[j + i * out->cols] += sum;
        }
    }
}

void _mat_mul_tn(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            f32 sum = 0.0f;
            for (u64 k = 0; k < a->rows; k++) {
                sum += a->data[i + k * a->cols] * b->data[j + k * b->cols];
            }
            out->data[j + i * out->cols] += sum;
        }
    }
}

void _mat_mul_tt(matrix* out, const matrix* a, const matrix* b) {
    for (u64 i = 0; i < out->rows; i++) {
        for (u64 j = 0; j < out->cols; j++) {
            f32 sum = 0.0f;
            for (u64 k = 0; k < a->rows; k++) {
                sum += a->data[i + k * a->cols] * b->data[k + j * b->cols];
            }
            out->data[j + i * out->cols] += sum;
        }
    }
}

b32 matmul(matrix* out, const matrix* a, const matrix* b, b8 zero_out, b8 transpose_a, b8 transpose_b) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;

    if (a_cols != b_rows || out->rows != a_rows || out->cols != b_cols) return false;
    if (zero_out) clear(out);

    u32 transpose = (transpose_a << 1) | transpose_b;
    switch (transpose) {
        case 0: _mat_mul_nn(out, a, b); break;
        case 1: _mat_mul_nt(out, a, b); break;
        case 2: _mat_mul_tn(out, a, b); break;
        case 3: _mat_mul_tt(out, a, b); break;
    }
    return true;
}

b32 relu(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = MAX(0.0f, in->data[i]);
    return true;
}

b32 softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    u64 size = (u64)out->rows * out->cols;

    f32 max_value = in->data[0];
    for (u64 i = 1; i < size; i++) max_value = MAX(max_value, in->data[i]);

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out->data[i] = expf(in->data[i] - max_value);
        sum += out->data[i];
    }
    scale(out, 1.0f / sum);
    return true;
}

b32 reinforce_loss(matrix* out, const matrix* probs, const matrix* advantages) {
    if (probs->rows != advantages->rows || probs->cols != advantages->cols) return false;
    if (out->rows != advantages->rows || out->cols != advantages->cols) return false;

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        f32 p = MAX(probs->data[i], 1e-8f);
        out->data[i] = -logf(p) * advantages->data[i];
    }
    return true;
}

b32 reinforce_add_grad(matrix* prob_grads, const matrix* probs, const matrix* rt, const matrix* grad) {
    if (probs->rows != rt->rows || probs->cols != rt->cols) return false;
    if (grad->rows != probs->rows || grad->cols != probs->cols) return false;
    if (prob_grads->rows != probs->rows || prob_grads->cols != probs->cols) return false;

    u64 size = (u64)probs->rows * probs->cols;
    for (u64 i = 0; i < size; i++) {
        f32 p = MAX(probs->data[i], 1e-8f);
        prob_grads->data[i] += grad->data[i] * (-rt->data[i] / p);
    }
    return true;
}

b32 relu_add_grad(matrix* out, const matrix* in, const matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    if (out->rows != grad->rows || out->cols != grad->cols) return false;

    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) {
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    }
    return true;
}

b32 softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad) {
    if (out->rows != softmax_out->rows || out->cols != softmax_out->cols) return false;
    if (grad->rows != softmax_out->rows || grad->cols != softmax_out->cols) return false;

    u64 size = (u64)softmax_out->rows * softmax_out->cols;
    f32 dot = 0.0f;
    for (u64 i = 0; i < size; i++) dot += grad->data[i] * softmax_out->data[i];

    for (u64 i = 0; i < size; i++) {
        out->data[i] += softmax_out->data[i] * (grad->data[i] - dot);
    }
    return true;
}

// ============================================================================
// AUTOGRAD COMPUTATIONAL GRAPH
// ============================================================================
typedef enum {
    VAR_FLAG_NONE = 0,
    VAR_FLAG_REQUIRES_GRAD  = (1 << 0),
    VAR_FLAG_PARAMETER      = (1 << 1),
} VarFlags;

#define VAR_MAX_INPUTS 2

typedef struct Var Var;
typedef struct VarType VarType;
typedef void VarFunc(Var* var);
typedef b32 VarShapeFunc(Var* a, Var* b, u32* rows, u32* cols);

struct VarType {
    const char* name;
    u32 num_inputs;
    VarShapeFunc* shape;
    VarFunc* forward;
    VarFunc* backward;
};

struct Var {
    u32 index;
    u32 flags;
    matrix* val;
    matrix* grad;
    const VarType* type;
    Var* inputs[VAR_MAX_INPUTS];
};

typedef struct {
    Var** vars;
    u32 size;
} Graph;

typedef struct {
    u32 num_vars;
    Var* input;
    Var* output;
    Var* advantage;
    Var* cost;

    Graph forward_graph;
    Graph cost_graph;
} model_state;

static void var_relu_forward(Var* var) { relu(var->val, var->inputs[0]->val); }
static void var_relu_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) {
        relu_add_grad(var->inputs[0]->grad, var->inputs[0]->val, var->grad);
    }
}

static void var_softmax_forward(Var* var) { softmax(var->val, var->inputs[0]->val); }
static void var_softmax_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) {
        softmax_add_grad(var->inputs[0]->grad, var->val, var->grad);
    }
}

static void var_add_forward(Var* var) { add(var->val, var->inputs[0]->val, var->inputs[1]->val); }
static void var_add_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) add(var->inputs[0]->grad, var->inputs[0]->grad, var->grad);
    if (var->inputs[1] && (var->inputs[1]->flags & VAR_FLAG_REQUIRES_GRAD)) add(var->inputs[1]->grad, var->inputs[1]->grad, var->grad);
}

static void var_sub_forward(Var* var) { sub(var->val, var->inputs[0]->val, var->inputs[1]->val); }
static void var_sub_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) add(var->inputs[0]->grad, var->inputs[0]->grad, var->grad);
    if (var->inputs[1] && (var->inputs[1]->flags & VAR_FLAG_REQUIRES_GRAD)) sub(var->inputs[1]->grad, var->inputs[1]->grad, var->grad);
}

static void var_matmul_forward(Var* var) { matmul(var->val, var->inputs[0]->val, var->inputs[1]->val, 1, 0, 0); }
static void var_matmul_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) matmul(var->inputs[0]->grad, var->grad, var->inputs[1]->val, 0, 0, 1);
    if (var->inputs[1] && (var->inputs[1]->flags & VAR_FLAG_REQUIRES_GRAD)) matmul(var->inputs[1]->grad, var->inputs[0]->val, var->grad, 0, 1, 0);
}

static void var_reinforce_forward(Var* var) { reinforce_loss(var->val, var->inputs[0]->val, var->inputs[1]->val); }
static void var_reinforce_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) {
        reinforce_add_grad(var->inputs[0]->grad, var->inputs[0]->val, var->inputs[1]->val, var->grad);
    }
}

static b32 var_shape_same(Var* a, Var* b, u32* rows, u32* cols) {
    if (a == NULL) return false;
    if (b != NULL && (a->val->rows != b->val->rows || a->val->cols != b->val->cols)) return false;
    *rows = a->val->rows; *cols = a->val->cols;
    return true;
}

static b32 var_shape_matmul(Var* a, Var* b, u32* rows, u32* cols) {
    if (a == NULL || b == NULL || a->val->cols != b->val->rows) return false;
    *rows = a->val->rows; *cols = b->val->cols;
    return true;
}

const VarType VAR_TYPE_RELU = { "relu", 1, var_shape_same, var_relu_forward, var_relu_backward };
const VarType VAR_TYPE_SOFTMAX = { "softmax", 1, var_shape_same, var_softmax_forward, var_softmax_backward };
const VarType VAR_TYPE_ADD = { "add", 2, var_shape_same, var_add_forward, var_add_backward };
const VarType VAR_TYPE_SUB = { "sub", 2, var_shape_same, var_sub_forward, var_sub_backward };
const VarType VAR_TYPE_MATMUL = { "matmul", 2, var_shape_matmul, var_matmul_forward, var_matmul_backward };
const VarType VAR_TYPE_REINFORCE_LOSS = { "reinforce_loss", 2, var_shape_same, var_reinforce_forward, var_reinforce_backward };

Var* var_create(mem_arena* arena, model_state* model, u32 rows, u32 cols, u32 flags) {
    Var* out = PUSH_STRUCT(arena, Var);
    out->index = model->num_vars++;
    out->flags = flags;
    out->type = NULL;
    out->val = create_matrix(arena, rows, cols);
    if (flags & VAR_FLAG_REQUIRES_GRAD) out->grad = create_matrix(arena, rows, cols);
    return out;
}

static Var* create_node(mem_arena* arena, model_state* model, const VarType* type, Var* a, Var* b) {
    if (type == NULL || type->shape == NULL) return NULL;

    u32 flags = VAR_FLAG_NONE;
    if ((a != NULL && (a->flags & VAR_FLAG_REQUIRES_GRAD)) || (b != NULL && (b->flags & VAR_FLAG_REQUIRES_GRAD))) {
        flags = VAR_FLAG_REQUIRES_GRAD;
    }

    u32 rows = 0, cols = 0;
    if (!type->shape(a, b, &rows, &cols)) return NULL;

    Var* out = var_create(arena, model, rows, cols, flags);
    out->type = type;
    out->inputs[0] = a;
    if (type->num_inputs > 1) out->inputs[1] = b;
    return out;
}

Var* var_relu(mem_arena* arena, model_state* model, Var* input) { return create_node(arena, model, &VAR_TYPE_RELU, input, NULL); }
Var* var_softmax(mem_arena* arena, model_state* model, Var* input) { return create_node(arena, model, &VAR_TYPE_SOFTMAX, input, NULL); }
Var* var_add(mem_arena* arena, model_state* model, Var* a, Var* b) { return create_node(arena, model, &VAR_TYPE_ADD, a, b); }
Var* var_sub(mem_arena* arena, model_state* model, Var* a, Var* b) { return create_node(arena, model, &VAR_TYPE_SUB, a, b); }
Var* var_matmul(mem_arena* arena, model_state* model, Var* a, Var* b) { return create_node(arena, model, &VAR_TYPE_MATMUL, a, b); }
Var* var_reinforce_loss(mem_arena* arena, model_state* model, Var* probs, Var* rt) { return create_node(arena, model, &VAR_TYPE_REINFORCE_LOSS, probs, rt); }

Graph build_graph(mem_arena* arena, model_state* model, Var* out_var) {
    Graph graph = { 0 };
    if (arena == NULL || model == NULL || out_var == NULL || model->num_vars == 0) return graph;

    b8* visited = calloc(model->num_vars, sizeof(*visited));
    Var** stack = malloc(sizeof(*stack) * model->num_vars);
    Var** out = malloc(sizeof(*out) * model->num_vars);

    if (!visited || !stack || !out) { free(visited); free(stack); free(out); return graph; }

    u32 stack_size = 0, out_size = 0;
    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        Var* cur = stack[--stack_size];
        if (cur->index >= model->num_vars) continue;

        if (visited[cur->index]) {
            if (out_size < model->num_vars) out[out_size++] = cur;
            continue;
        }

        visited[cur->index] = true;
        if (stack_size < model->num_vars) stack[stack_size++] = cur;

        u32 num_inputs = cur->type != NULL ? cur->type->num_inputs : 0;
        for (u32 i = 0; i < num_inputs; i++) {
            Var* input = cur->inputs[i];
            if (input->index >= model->num_vars || visited[input->index]) continue;

            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == input) {
                    for (u32 k = j; k < stack_size - 1; k++) stack[k] = stack[k + 1];
                    stack_size--;
                }
            }
            if (stack_size < model->num_vars) stack[stack_size++] = input;
        }
    }

    graph.vars = PUSH_ARRAY_UNINIT(arena, Var*, out_size);
    if (graph.vars != NULL) {
        graph.size = out_size;
        memcpy(graph.vars, out, sizeof(*out) * out_size);
    }

    free(visited); free(stack); free(out);
    return graph;
}

void forward_pass(Graph* graph) {
    for (u32 i = 0; i < graph->size; i++) {
        Var* cur = graph->vars[i];
        if (cur->type && cur->type->forward) cur->type->forward(cur);
    }
}

void backward_pass(Graph* graph) {
    for (u32 i = 0; i < graph->size; i++) {
        Var* cur = graph->vars[i];
        if (!(cur->flags & VAR_FLAG_REQUIRES_GRAD) || (cur->flags & VAR_FLAG_PARAMETER)) continue;
        clear(cur->grad);
    }

    fill(graph->vars[graph->size - 1]->grad, 1.0f);

    for (i64 i = (i64)graph->size - 1; i >= 0; i--) {
        Var* cur = graph->vars[i];
        if (!(cur->flags & VAR_FLAG_REQUIRES_GRAD)) continue;
        if (cur->type && cur->type->backward) cur->type->backward(cur);
    }
}

// ============================================================================
// POLICY MODEL CREATION
// ============================================================================
void create_actor_model(mem_arena* arena, model_state* model) {
    Var* input = var_create(arena, model, 76, 1, VAR_FLAG_NONE);
    model->input = input;

    Var* W0 = var_create(arena, model, 128, 76, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b0 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W1 = var_create(arena, model, 128, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b1 = var_create(arena, model, 128, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W2 = var_create(arena, model, 5, 128, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b2 = var_create(arena, model, 5, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    f32 bound0 = sqrtf(6.0f / (76 + 128));
    f32 bound1 = sqrtf(6.0f / (128 + 128));
    f32 bound2 = sqrtf(6.0f / (128 + 5));

    fill_rand(W0->val, -bound0, bound0);
    fill_rand(W1->val, -bound1, bound1);
    fill_rand(W2->val, -bound2, bound2);

    Var* z0_a = var_matmul(arena, model, W0, input);
    Var* z0_b = var_add(arena, model, z0_a, b0);
    Var* a0 = var_relu(arena, model, z0_b);

    Var* z1_a = var_matmul(arena, model, W1, a0);
    Var* z1_b = var_add(arena, model, z1_a, b1);
    Var* a1 = var_relu(arena, model, z1_b);

    Var* z2_a = var_matmul(arena, model, W2, a1);
    Var* z2_b = var_add(arena, model, z2_a, b2);

    Var* output = var_softmax(arena, model, z2_b);
    model->output = output;

    Var* advantage = var_create(arena, model, 5, 1, VAR_FLAG_NONE);
    model->advantage = advantage;

    Var* cost = var_reinforce_loss(arena, model, output, advantage);
    model->cost = cost;
}

// ============================================================================
// ENVIRONMENT & REINFORCE TRAINING LOOP
// ============================================================================
typedef enum { LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3, NONE = 4 } ACTION;
typedef struct { i32 x, y; } State;

typedef struct {
    State snake;
    State food;
    f32 score;
    u32 foods_eaten;
    u32 rows, cols, grid_size;
    u64 steps;
    ACTION pov;
} SnakeENV;

#define BUFFER_SIZE 64
#define EPISODE_LEN 100

typedef struct {
    State states[EPISODE_LEN];
    State food_states[EPISODE_LEN];
    ACTION povs[EPISODE_LEN];
    ACTION actions[EPISODE_LEN];
    f32 rewards[EPISODE_LEN];
    f32 returns[EPISODE_LEN];
    State next_states[EPISODE_LEN];
    b32 dones[EPISODE_LEN];
    u32 len;
} Trajectory;

typedef struct {
    Trajectory trajectories[BUFFER_SIZE];
    u32 count;
} ReplayBuffer;

SnakeENV* create_env(u32 grid_size) {
    SnakeENV* env = malloc(sizeof(*env));
    if (!env) return NULL;

    u32 side = (u32)sqrtf((f32)grid_size);
    if (side * side != grid_size) { free(env); return NULL; }

    env->snake = (State){ .x = 0, .y = 0 };
    env->food = (State){ .x = 5, .y = 5 };
    env->rows = side; env->cols = side; env->grid_size = grid_size;
    env->score = 0.0f; env->foods_eaten = 0; env->pov = RIGHT; env->steps = 0;
    return env;
}

State get_random_food_loc(u32 cols, u32 rows) {
    return (State) { .x = (i32)randn(cols), .y = (i32)randn(rows) };
}

b32 game_over(SnakeENV* env) {
    return env->snake.x < 0 || env->snake.x >= (i32)env->cols ||
           env->snake.y < 0 || env->snake.y >= (i32)env->rows;
}

f32 get_reward(SnakeENV* env) {
    f32 reward = 0.0f;
    if (env->snake.x == env->food.x && env->snake.y == env->food.y) {
        reward += 20.0f;
        env->foods_eaten++;
        State new_food;
        do {
            new_food = get_random_food_loc(env->cols, env->rows);
        } while (new_food.x == env->snake.x && new_food.y == env->snake.y);
        env->food = new_food;
    }
    if (env->snake.x < 0 || env->snake.x >= (i32)env->cols ||
        env->snake.y < 0 || env->snake.y >= (i32)env->rows) {
        reward -= 10.0f;
    }
    return reward;
}

void reset_state(SnakeENV* env) {
    env->snake = (State){ .x = (i32)(env->cols / 2), .y = (i32)(env->rows / 2) };
    do {
        env->food = get_random_food_loc(env->cols, env->rows);
    } while (env->food.x == env->snake.x && env->food.y == env->snake.y);
    env->score = 0.0f; env->foods_eaten = 0; env->pov = RIGHT; env->steps = 0;
}

void take_action(SnakeENV* env, ACTION action) {
    if (action == NONE) action = env->pov;
    if (action == LEFT) env->snake.x -= 1;
    else if (action == RIGHT) env->snake.x += 1;
    else if (action == UP) env->snake.y += 1;
    else if (action == DOWN) env->snake.y -= 1;
    else return;
    env->pov = action;
}

ACTION sample_action(matrix* probs) {
    f32 r = prng_randf();
    f32 cdf = 0.0f;
    u32 size = probs->rows * probs->cols;
    for (u32 i = 0; i < size; i++) {
        cdf += probs->data[i];
        if (r <= cdf) return (ACTION)i;
    }
    return (ACTION)(size - 1);
}

void build_state_vector(matrix* in, State state, State food_state, ACTION pov, u32 cols, u32 grid_size) {
    clear(in);
    u32 snake_i = (u32)state.y * cols + (u32)state.x;
    u32 food_i = (u32)food_state.y * cols + (u32)food_state.x;
    in->data[snake_i] = 1.0f;
    in->data[grid_size + food_i] = 1.0f;
    in->data[2 * grid_size + (u32)pov] = 1.0f;
}

void train(model_state* model, SnakeENV* env, mem_arena* arena) {
    (void)arena;
    u32 EPOCHS = 2000;
    u32 rollout_size = BUFFER_SIZE;
    u32 episode_len = EPISODE_LEN;
    f32 gamma = 0.99f;
    f32 learning_rate = 0.05f;

    ReplayBuffer* buffer = calloc(1, sizeof(ReplayBuffer));
    if (!buffer) return;

    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        u32 foods_eaten = 0;

        for (u32 i = 0; i < rollout_size; i++) {
            reset_state(env);
            Trajectory* traj = &buffer->trajectories[i];
            traj->len = 0;

            for (u32 t = 0; t < episode_len; t++) {
                State state = env->snake;
                State food_state = env->food;
                ACTION pov = env->pov;

                build_state_vector(model->input->val, state, food_state, pov, env->cols, env->grid_size);
                forward_pass(&model->forward_graph);
                ACTION action = sample_action(model->output->val);

                take_action(env, action);

                f32 reward = get_reward(env);
                env->score += reward;
                b32 done = game_over(env);

                traj->states[t] = state;
                traj->food_states[t] = food_state;
                traj->povs[t] = pov;
                traj->actions[t] = action;
                traj->rewards[t] = reward;
                traj->next_states[t] = env->snake;
                traj->dones[t] = done;
                traj->len = t + 1;

                if (done) break;
            }
            foods_eaten += env->foods_eaten;
        }

        buffer->count = rollout_size;
        u32 sample_count = 0;
        f32 average_return = 0.0f, return_sum = 0.0f, return_sq_sum = 0.0f;

        for (u32 b = 0; b < buffer->count; b++) {
            Trajectory* traj = &buffer->trajectories[b];
            f32 G = 0.0f;
            for (i32 t = (i32)traj->len - 1; t >= 0; t--) {
                G = traj->rewards[t] + gamma * G;
                traj->returns[t] = G;
                return_sum += G;
                return_sq_sum += G * G;
                sample_count++;
            }
            average_return += traj->returns[0];
        }

        f32 return_mean = return_sum / (f32)sample_count;
        f32 return_variance = return_sq_sum / (f32)sample_count - return_mean * return_mean;
        f32 return_std = sqrtf(MAX(return_variance, 0.0f));

        for (u32 i = 0; i < model->cost_graph.size; i++) {
            Var* cur = model->cost_graph.vars[i];
            if (cur->flags & VAR_FLAG_PARAMETER) clear(cur->grad);
        }

        for (u32 b = 0; b < buffer->count; b++) {
            Trajectory* traj = &buffer->trajectories[b];
            for (u32 t = 0; t < traj->len; t++) {
                build_state_vector(model->input->val, traj->states[t], traj->food_states[t], traj->povs[t], env->cols, env->grid_size);

                clear(model->advantage->val);
                f32 advantage = (traj->returns[t] - return_mean) / (return_std + 1e-8f);
                model->advantage->val->data[traj->actions[t]] = advantage;

                forward_pass(&model->cost_graph);
                backward_pass(&model->cost_graph);
            }
        }

        if (sample_count > 0) {
            f32 gradient_scale = learning_rate / (f32)sample_count;
            for (u32 i = 0; i < model->cost_graph.size; i++) {
                Var* cur = model->cost_graph.vars[i];
                if (!(cur->flags & VAR_FLAG_PARAMETER)) continue;
                scale(cur->grad, gradient_scale);
                sub(cur->val, cur->val, cur->grad);
            }
        }

        printf("Epoch %u | Average return: %.3f | Foods: %u | Samples: %u | Return std: %.3f\n",
               epoch, average_return / (f32)buffer->count, foods_eaten, sample_count, return_std);
        fflush(stdout);
    }

    free(buffer);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf(">>> Program started. If you can read this line, your console is working. <<<\n");
    fflush(stdout);

    mem_arena* arena = arena_create(GiB(1));
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

    create_actor_model(arena, model);
    model->forward_graph = build_graph(arena, model, model->output);
    model->cost_graph = build_graph(arena, model, model->cost);

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
