#include "model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// AUTOGRAD COMPUTATIONAL GRAPH
// ============================================================================
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

// --- Plain elementwise add (equal-shape inputs) ---
static void var_add_forward(Var* var) { add(var->val, var->inputs[0]->val, var->inputs[1]->val); }
static void var_add_backward(Var* var) {
    if (var->inputs[0] && (var->inputs[0]->flags & VAR_FLAG_REQUIRES_GRAD)) add(var->inputs[0]->grad, var->inputs[0]->grad, var->grad);
    if (var->inputs[1] && (var->inputs[1]->flags & VAR_FLAG_REQUIRES_GRAD)) add(var->inputs[1]->grad, var->inputs[1]->grad, var->grad);
}

// --- Bias add: broadcasts a (rows,1) bias across a (rows,BATCH) input ---
static void var_bias_add_forward(Var* var) {
    matrix* a = var->inputs[0]->val;
    matrix* bias = var->inputs[1]->val;
    for (u32 i = 0; i < a->rows; i++) {
        f32 bi = bias->data[i];
        f32* out_row = var->val->data + (u64)i * a->cols;
        const f32* a_row = a->data + (u64)i * a->cols;
        for (u32 j = 0; j < a->cols; j++)
            out_row[j] = a_row[j] + bi;
    }
}

static void var_bias_add_backward(Var* var) {
    Var* a = var->inputs[0];
    Var* bias = var->inputs[1];
    matrix* grad = var->grad;

    if (a->flags & VAR_FLAG_REQUIRES_GRAD) add(a->grad, a->grad, grad);

    if (bias->flags & VAR_FLAG_REQUIRES_GRAD) {
        for (u32 i = 0; i < grad->rows; i++) {
            f32 sum = 0.0f;
            const f32* grad_row = grad->data + (u64)i * grad->cols;
            for (u32 j = 0; j < grad->cols; j++) sum += grad_row[j];
            bias->grad->data[i] += sum;
        }
    }
}

static b32 var_shape_bias_add(Var* a, Var* b, u32* rows, u32* cols) {
    if (a == NULL || b == NULL || b->val->cols != 1 || b->val->rows != a->val->rows) return false;
    *rows = a->val->rows; *cols = a->val->cols;
    return true;
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
const VarType VAR_TYPE_BIAS_ADD = { "bias_add", 2, var_shape_bias_add, var_bias_add_forward, var_bias_add_backward };
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
Var* var_bias_add(mem_arena* arena, model_state* model, Var* a, Var* b) { return create_node(arena, model, &VAR_TYPE_BIAS_ADD, a, b); }
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
void create_actor_model(mem_arena* arena, model_state* model, u32 batch) {
    Var* input = var_create(arena, model, OBSERVATION_SIZE, batch, VAR_FLAG_NONE);
    model->input = input;

    Var* W0 = var_create(arena, model, HIDDEN_SIZE, OBSERVATION_SIZE, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b0 = var_create(arena, model, HIDDEN_SIZE, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W1 = var_create(arena, model, HIDDEN_SIZE, HIDDEN_SIZE, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b1 = var_create(arena, model, HIDDEN_SIZE, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    Var* W2 = var_create(arena, model, 5, HIDDEN_SIZE, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);
    Var* b2 = var_create(arena, model, 5, 1, VAR_FLAG_PARAMETER | VAR_FLAG_REQUIRES_GRAD);

    f32 bound0 = sqrtf(6.0f / (OBSERVATION_SIZE + HIDDEN_SIZE));
    f32 bound1 = sqrtf(6.0f / (HIDDEN_SIZE + HIDDEN_SIZE));
    f32 bound2 = sqrtf(6.0f / (HIDDEN_SIZE + 5));

    fill_rand(W0->val, -bound0, bound0);
    fill_rand(W1->val, -bound1, bound1);
    fill_rand(W2->val, -bound2, bound2);

    Var* z0_a = var_matmul(arena, model, W0, input);
    Var* z0_b = var_bias_add(arena, model, z0_a, b0);
    Var* a0 = var_relu(arena, model, z0_b);

    Var* z1_a = var_matmul(arena, model, W1, a0);
    Var* z1_b = var_bias_add(arena, model, z1_a, b1);
    Var* a1 = var_relu(arena, model, z1_b);

    Var* z2_a = var_matmul(arena, model, W2, a1);
    Var* z2_b = var_bias_add(arena, model, z2_a, b2);

    Var* output = var_softmax(arena, model, z2_b);
    model->output = output;

    Var* advantage = var_create(arena, model, 5, batch, VAR_FLAG_NONE);
    model->advantage = advantage;

    Var* cost = var_reinforce_loss(arena, model, output, advantage);
    model->cost = cost;
}

// ============================================================================
