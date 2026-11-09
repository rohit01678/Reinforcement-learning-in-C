#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"

typedef enum {
    VAR_FLAG_NONE = 0,
    VAR_FLAG_REQUIRES_GRAD = (1 << 0),
    VAR_FLAG_PARAMETER = (1 << 1),
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

Var* var_create(mem_arena* arena, model_state* model, u32 rows, u32 cols, u32 flags);
Var* var_relu(mem_arena* arena, model_state* model, Var* input);
Var* var_softmax(mem_arena* arena, model_state* model, Var* input);
Var* var_add(mem_arena* arena, model_state* model, Var* a, Var* b);
Var* var_bias_add(mem_arena* arena, model_state* model, Var* a, Var* b);
Var* var_sub(mem_arena* arena, model_state* model, Var* a, Var* b);
Var* var_matmul(mem_arena* arena, model_state* model, Var* a, Var* b);
Var* var_reinforce_loss(mem_arena* arena, model_state* model, Var* probs, Var* returns);
Graph build_graph(mem_arena* arena, model_state* model, Var* out_var);
void forward_pass(Graph* graph);
void backward_pass(Graph* graph);
void create_actor_model(mem_arena* arena, model_state* model, u32 batch);

#endif