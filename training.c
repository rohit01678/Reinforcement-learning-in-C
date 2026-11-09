#include "training.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// REINFORCE TRAINING LOOP
// ============================================================================
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

void train(model_state* model, SnakeENV* env, mem_arena* arena) {
    (void)arena;
    u32 EPOCHS = 2000;
    u32 episode_len = EPISODE_LEN;
    f32 gamma = 0.99f;
    f32 learning_rate = 0.05f;

    ReplayBuffer* buffer = calloc(1, sizeof(ReplayBuffer));
    BatchEnv* benv = malloc(sizeof(BatchEnv));
    if (!buffer || !benv) {
        fprintf(stderr, "Failed to allocate training buffers!\n");
        free(buffer);
        free(benv);
        return;
    }

    // Reused scratch space for the flattened per-epoch training set.
    // Sized for the worst case: every trajectory runs the full episode.
    u32 max_samples = BUFFER_SIZE * episode_len;
    State* flat_state = malloc(sizeof(State) * max_samples);
    State* flat_food = malloc(sizeof(State) * max_samples);
    ACTION* flat_pov = malloc(sizeof(ACTION) * max_samples);
    ACTION* flat_action = malloc(sizeof(ACTION) * max_samples);
    f32* flat_advantage = malloc(sizeof(f32) * max_samples);

    if (!flat_state || !flat_food || !flat_pov || !flat_action || !flat_advantage) {
        fprintf(stderr, "Failed to allocate flattened sample buffers!\n");
        free(buffer); free(benv);
        free(flat_state); free(flat_food); free(flat_pov); free(flat_action); free(flat_advantage);
        return;
    }

    for (u32 epoch = 0; epoch < EPOCHS; epoch++) {
        // ------------------------------------------------------------
        // ROLLOUT: step all BATCH trajectories together at each
        // timestep, doing exactly one forward pass per timestep
        // instead of one per (trajectory, timestep) pair.
        // ------------------------------------------------------------
        reset_batch_env(benv, env);
        for (u32 i = 0; i < BUFFER_SIZE; i++) buffer->trajectories[i].len = 0;

        for (u32 t = 0; t < episode_len; t++) {
            b32 any_active = false;
            clear(model->input->val);
            for (u32 i = 0; i < BATCH; i++) {
                if (!benv->active[i]) continue;
                any_active = true;
                build_state_vector_col(model->input->val, i, benv->snake[i], benv->food[i], benv->pov[i], env->cols, env->grid_size);
            }
            if (!any_active) break;

            forward_pass(&model->forward_graph);

            for (u32 i = 0; i < BATCH; i++) {
                if (!benv->active[i]) continue;

                Trajectory* traj = &buffer->trajectories[i];
                State state = benv->snake[i];
                State food_state = benv->food[i];
                ACTION pov = benv->pov[i];

                ACTION action = sample_action_col(model->output->val, i);
                f32 reward = step_batch_env(benv, i, env, action);
                b32 done = state_out_of_bounds(benv->snake[i], env->cols, env->rows);

                traj->states[t] = state;
                traj->food_states[t] = food_state;
                traj->povs[t] = pov;
                traj->actions[t] = action;
                traj->rewards[t] = reward;
                traj->next_states[t] = benv->snake[i];
                traj->dones[t] = done;
                traj->len = t + 1;

                if (done) benv->active[i] = false;
            }
        }

        u32 foods_eaten = 0;
        for (u32 i = 0; i < BATCH; i++) foods_eaten += benv->foods_eaten[i];

        // ------------------------------------------------------------
        // RETURNS: unchanged, cheap sequential pass per trajectory.
        // ------------------------------------------------------------
        buffer->count = BUFFER_SIZE;
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

        // ------------------------------------------------------------
        // FLATTEN: gather all transitions from all trajectories into
        // contiguous arrays so they can be processed in batches.
        // ------------------------------------------------------------
        u32 idx = 0;
        for (u32 b = 0; b < buffer->count; b++) {
            Trajectory* traj = &buffer->trajectories[b];
            for (u32 t = 0; t < traj->len; t++) {
                flat_state[idx] = traj->states[t];
                flat_food[idx] = traj->food_states[t];
                flat_pov[idx] = traj->povs[t];
                flat_action[idx] = traj->actions[t];
                flat_advantage[idx] = (traj->returns[t] - return_mean) / (return_std + 1e-8f);
                idx++;
            }
        }

        // ------------------------------------------------------------
        // TRAIN: process the flattened samples BATCH at a time.
        // Gradients accumulate across chunks exactly as they did
        // across individual samples before (backward_pass only
        // clears non-parameter grads, never parameter grads).
        // The last, possibly-partial chunk is zero-padded: padded
        // columns have advantage == 0, so reinforce_add_grad
        // contributes exactly zero gradient for them.
        // ------------------------------------------------------------
        for (u32 i = 0; i < model->cost_graph.size; i++) {
            Var* cur = model->cost_graph.vars[i];
            if (cur->flags & VAR_FLAG_PARAMETER) clear(cur->grad);
        }

        for (u32 start = 0; start < sample_count; start += BATCH) {
            u32 n = MIN(BATCH, sample_count - start);

            clear(model->input->val);
            clear(model->advantage->val);
            for (u32 c = 0; c < n; c++) {
                u32 s = start + c;
                build_state_vector_col(model->input->val, c, flat_state[s], flat_food[s], flat_pov[s], env->cols, env->grid_size);
                model->advantage->val->data[c + (u64)flat_action[s] * model->advantage->val->cols] = flat_advantage[s];
            }

            forward_pass(&model->cost_graph);
            backward_pass(&model->cost_graph);
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

    free(flat_state); free(flat_food); free(flat_pov); free(flat_action); free(flat_advantage);
    free(benv);
    free(buffer);
}

// ============================================================================
// MAIN ENTRY POINT
// ============================================================================
