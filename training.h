#ifndef TRAINING_H
#define TRAINING_H

#include "environment.h"
#include "model.h"

void train(model_state* model, SnakeENV* env, mem_arena* arena);

#endif