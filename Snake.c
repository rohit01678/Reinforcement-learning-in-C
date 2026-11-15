#define _CRT_SECURE_NO_WARNINGS
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REPLAY_CAPACITY 2048

#include "arena.h"
#include "environment.h"
#include "model.h"
#include "training.h"

static u32 graph_parameter_count(const Graph* graph) {
    u32 count = 0;
    for (u32 i = 0; i < graph->size; i++) count += (graph->vars[i]->flags & VAR_FLAG_PARAMETER) != 0;
    return count;
}

static u64 graph_parameter_values(const Graph* graph) {
    u64 count = 0;
    for (u32 i = 0; i < graph->size; i++) {
        Var* var = graph->vars[i];
        if (var->flags & VAR_FLAG_PARAMETER) count += (u64)var->val->rows * var->val->cols;
    }
    return count;
}

static b32 save_policy(const char* path, const model_state* model) {
    static const char magic[8] = { 'S', 'N', 'A', 'K', 'E', 'R', 'L', '1' };
    u32 version = 1;
    u32 grid_size = GRID_SIZE;
    u32 observation_size = OBSERVATION_SIZE;
    u32 hidden_size = HIDDEN_SIZE;
    u32 parameter_count = graph_parameter_count(&model->cost_graph);
    FILE* file = fopen(path, "wb");
    if (!file) return false;

    b32 success = fwrite(magic, sizeof(magic), 1, file) == 1 &&
                  fwrite(&version, sizeof(version), 1, file) == 1 &&
                  fwrite(&grid_size, sizeof(grid_size), 1, file) == 1 &&
                  fwrite(&observation_size, sizeof(observation_size), 1, file) == 1 &&
                  fwrite(&hidden_size, sizeof(hidden_size), 1, file) == 1 &&
                  fwrite(&parameter_count, sizeof(parameter_count), 1, file) == 1;
    for (u32 i = 0; success && i < model->cost_graph.size; i++) {
        Var* var = model->cost_graph.vars[i];
        if (!(var->flags & VAR_FLAG_PARAMETER)) continue;
        u32 rows = var->val->rows;
        u32 cols = var->val->cols;
        u64 value_count = (u64)rows * cols;
        success = fwrite(&rows, sizeof(rows), 1, file) == 1 &&
                  fwrite(&cols, sizeof(cols), 1, file) == 1 &&
                  fwrite(var->val->data, sizeof(f32), value_count, file) == value_count;
    }
    if (fclose(file) != 0) success = false;
    return success;
}

static b32 load_policy(const char* path, model_state* model) {
    static const char magic[8] = { 'S', 'N', 'A', 'K', 'E', 'R', 'L', '1' };
    FILE* file = fopen(path, "rb");
    if (!file) return false;

    char file_magic[sizeof(magic)];
    u32 version = 0, grid_size = 0, observation_size = 0, hidden_size = 0, parameter_count = 0;
    b32 valid = fread(file_magic, sizeof(file_magic), 1, file) == 1 &&
                fread(&version, sizeof(version), 1, file) == 1 &&
                fread(&grid_size, sizeof(grid_size), 1, file) == 1 &&
                fread(&observation_size, sizeof(observation_size), 1, file) == 1 &&
                fread(&hidden_size, sizeof(hidden_size), 1, file) == 1 &&
                fread(&parameter_count, sizeof(parameter_count), 1, file) == 1 &&
                memcmp(file_magic, magic, sizeof(magic)) == 0 && version == 1 &&
                grid_size == GRID_SIZE && observation_size == OBSERVATION_SIZE &&
                hidden_size == HIDDEN_SIZE && parameter_count == graph_parameter_count(&model->cost_graph);
    u64 total_values = graph_parameter_values(&model->cost_graph);
    f32* values = valid ? malloc(sizeof(f32) * total_values) : NULL;
    if (!values) valid = false;

    u64 offset = 0;
    for (u32 i = 0; valid && i < model->cost_graph.size; i++) {
        Var* var = model->cost_graph.vars[i];
        if (!(var->flags & VAR_FLAG_PARAMETER)) continue;
        u32 rows = 0, cols = 0;
        if (fread(&rows, sizeof(rows), 1, file) != 1 || fread(&cols, sizeof(cols), 1, file) != 1 ||
            rows != var->val->rows || cols != var->val->cols) {
            valid = false;
            break;
        }
        u64 value_count = (u64)rows * cols;
        if (fread(values + offset, sizeof(f32), value_count, file) != value_count) {
            valid = false;
            break;
        }
        offset += value_count;
    }
    if (valid && (fgetc(file) != EOF || ferror(file))) valid = false;
    fclose(file);

    if (valid) {
        offset = 0;
        for (u32 i = 0; i < model->cost_graph.size; i++) {
            Var* var = model->cost_graph.vars[i];
            if (!(var->flags & VAR_FLAG_PARAMETER)) continue;
            u64 value_count = (u64)var->val->rows * var->val->cols;
            memcpy(var->val->data, values + offset, sizeof(f32) * value_count);
            offset += value_count;
        }
    }
    free(values);
    return valid;
}

static b32 copy_policy_parameters(const model_state* source, model_state* destination) {
    u32 destination_index = 0;
    u32 copied = 0;
    for (u32 i = 0; i < source->forward_graph.size; i++) {
        Var* source_var = source->forward_graph.vars[i];
        if (!(source_var->flags & VAR_FLAG_PARAMETER)) continue;
        while (destination_index < destination->forward_graph.size &&
               !(destination->forward_graph.vars[destination_index]->flags & VAR_FLAG_PARAMETER)) destination_index++;
        if (destination_index >= destination->forward_graph.size) return false;
        Var* destination_var = destination->forward_graph.vars[destination_index++];
        if (source_var->val->rows != destination_var->val->rows || source_var->val->cols != destination_var->val->cols) return false;
        u64 value_count = (u64)source_var->val->rows * source_var->val->cols;
        memcpy(destination_var->val->data, source_var->val->data, sizeof(f32) * value_count);
        copied++;
    }
    return copied == graph_parameter_count(&destination->forward_graph);
}

typedef struct {
    f32 observation[OBSERVATION_SIZE];
    ACTION action;
    f32 reward;
} OnlineSample;

typedef struct {
    f32 observation[OBSERVATION_SIZE];
    ACTION action;
    f32 advantage;
} ReplaySample;

typedef struct {
    ReplaySample* samples;
    u32 count;
    u32 next_index;
} ReplayBuffer;

static b32 append_online_sample(OnlineSample** samples, u32* count, u32* capacity, const f32* observation, ACTION action, f32 reward) {
    if (*count == *capacity) {
        u32 new_capacity = *capacity ? *capacity * 2 : 256;
        if (new_capacity < *capacity || new_capacity == 0 ||
            sizeof(**samples) > SIZE_MAX / (size_t)new_capacity) return false;
        OnlineSample* new_samples = realloc(*samples, sizeof(**samples) * (size_t)new_capacity);
        if (!new_samples) return false;
        *samples = new_samples;
        *capacity = new_capacity;
    }

    OnlineSample* sample = &(*samples)[(*count)++];
    memcpy(sample->observation, observation, sizeof(sample->observation));
    sample->action = action;
    sample->reward = reward;
    return true;
}

static void append_replay_sample(ReplayBuffer* replay, const ReplaySample* sample) {
    if (replay->count < REPLAY_CAPACITY) {
        replay->samples[replay->count++] = *sample;
        replay->next_index = replay->count % REPLAY_CAPACITY;
    } else {
        replay->samples[replay->next_index] = *sample;
        replay->next_index = (replay->next_index + 1) % REPLAY_CAPACITY;
    }
}

static const ReplaySample* replay_sample_at(const ReplayBuffer* replay, u32 index) {
    u32 oldest = replay->count == REPLAY_CAPACITY ? replay->next_index : 0;
    return &replay->samples[(oldest + index) % REPLAY_CAPACITY];
}

static b32 load_replay(const char* path, ReplayBuffer* replay) {
    static const char magic[8] = { 'S', 'N', 'A', 'K', 'E', 'R', 'P', '1' };
    FILE* file = fopen(path, "rb");
    if (!file) {
        if (errno == ENOENT) return true;
        fprintf(stderr, "Could not open experience replay: %s\n", path);
        return false;
    }

    char file_magic[sizeof(magic)];
    u32 version = 0, observation_size = 0, capacity = 0, count = 0;
    b32 valid = fread(file_magic, sizeof(file_magic), 1, file) == 1 &&
                fread(&version, sizeof(version), 1, file) == 1 &&
                fread(&observation_size, sizeof(observation_size), 1, file) == 1 &&
                fread(&capacity, sizeof(capacity), 1, file) == 1 &&
                fread(&count, sizeof(count), 1, file) == 1 &&
                memcmp(file_magic, magic, sizeof(magic)) == 0 && version == 1 &&
                observation_size == OBSERVATION_SIZE && capacity == REPLAY_CAPACITY &&
                count <= REPLAY_CAPACITY;

    ReplaySample* loaded = valid ? malloc(sizeof(*loaded) * REPLAY_CAPACITY) : NULL;
    if (!loaded) valid = false;
    for (u32 i = 0; valid && i < count; i++) {
        u32 action = 0;
        valid = fread(loaded[i].observation, sizeof(f32), OBSERVATION_SIZE, file) == OBSERVATION_SIZE &&
                fread(&action, sizeof(action), 1, file) == 1 &&
                fread(&loaded[i].advantage, sizeof(f32), 1, file) == 1 &&
                action <= DOWN && isfinite(loaded[i].advantage);
        for (u32 j = 0; valid && j < OBSERVATION_SIZE; j++) {
            if (!isfinite(loaded[i].observation[j])) valid = false;
        }
        loaded[i].action = (ACTION)action;
    }
    if (valid && (fgetc(file) != EOF || ferror(file))) valid = false;
    fclose(file);

    if (!valid) {
        fprintf(stderr, "Ignoring invalid experience replay: %s\n", path);
        free(loaded);
        return true;
    }
    memcpy(replay->samples, loaded, sizeof(*loaded) * count);
    replay->count = count;
    replay->next_index = count % REPLAY_CAPACITY;
    free(loaded);
    fprintf(stderr, "Loaded %u experience replay samples.\n", replay->count);
    return true;
}

static b32 save_replay(const char* path, const ReplayBuffer* replay) {
    static const char magic[8] = { 'S', 'N', 'A', 'K', 'E', 'R', 'P', '1' };
    u32 version = 1;
    u32 observation_size = OBSERVATION_SIZE;
    u32 capacity = REPLAY_CAPACITY;
    FILE* file = fopen(path, "wb");
    if (!file) return false;

    b32 success = fwrite(magic, sizeof(magic), 1, file) == 1 &&
                  fwrite(&version, sizeof(version), 1, file) == 1 &&
                  fwrite(&observation_size, sizeof(observation_size), 1, file) == 1 &&
                  fwrite(&capacity, sizeof(capacity), 1, file) == 1 &&
                  fwrite(&replay->count, sizeof(replay->count), 1, file) == 1;
    for (u32 i = 0; success && i < replay->count; i++) {
        const ReplaySample* sample = replay_sample_at(replay, i);
        u32 action = (u32)sample->action;
        success = fwrite(sample->observation, sizeof(f32), OBSERVATION_SIZE, file) == OBSERVATION_SIZE &&
                  fwrite(&action, sizeof(action), 1, file) == 1 &&
                  fwrite(&sample->advantage, sizeof(f32), 1, file) == 1;
    }
    if (fclose(file) != 0) success = false;
    return success;
}

static b32 update_online_policy(model_state* model, const OnlineSample* samples, u32 sample_count,
                                const char* checkpoint_path, ReplayBuffer* replay, const char* replay_path) {
    if (sample_count == 0) return true;

    f32* returns = malloc(sizeof(*returns) * sample_count);
    if (!returns) return false;

    f32 total_return = 0.0f;
    f32 total_return_squared = 0.0f;
    f32 discounted_return = 0.0f;
    for (i64 i = (i64)sample_count - 1; i >= 0; i--) {
        discounted_return = samples[i].reward + 0.99f * discounted_return;
        returns[i] = discounted_return;
        total_return += discounted_return;
        total_return_squared += discounted_return * discounted_return;
    }
    f32 mean_return = total_return / (f32)sample_count;
    f32 variance = total_return_squared / (f32)sample_count - mean_return * mean_return;
    f32 std_return = sqrtf(MAX(variance, 0.0f));

    if (replay) {
        for (u32 i = 0; i < sample_count; i++) {
            ReplaySample replay_sample = { 0 };
            memcpy(replay_sample.observation, samples[i].observation, sizeof(replay_sample.observation));
            replay_sample.action = samples[i].action;
            replay_sample.advantage = std_return > 1e-6f ? (returns[i] - mean_return) / std_return : returns[i];
            append_replay_sample(replay, &replay_sample);
        }
        if (!save_replay(replay_path, replay)) {
            free(returns);
            return false;
        }
    }

    u32 training_count = replay ? replay->count : sample_count;
    for (u32 i = 0; i < model->cost_graph.size; i++) {
        Var* var = model->cost_graph.vars[i];
        if (var->flags & VAR_FLAG_PARAMETER) clear(var->grad);
    }

    for (u32 i = 0; i < training_count; i++) {
        const f32* observation;
        ACTION action;
        f32 advantage;
        if (replay) {
            const ReplaySample* replay_sample = replay_sample_at(replay, i);
            observation = replay_sample->observation;
            action = replay_sample->action;
            advantage = replay_sample->advantage;
        } else {
            observation = samples[i].observation;
            action = samples[i].action;
            advantage = std_return > 1e-6f ? (returns[i] - mean_return) / std_return : returns[i];
        }
        memcpy(model->input->val->data, observation, sizeof(samples[i].observation));
        clear(model->advantage->val);
        model->advantage->val->data[(u64)action * model->advantage->val->cols] = advantage;
        forward_pass(&model->cost_graph);
        backward_pass(&model->cost_graph);
    }

    f32 gradient_scale = 0.01f / (f32)training_count;
    for (u32 i = 0; i < model->cost_graph.size; i++) {
        Var* var = model->cost_graph.vars[i];
        if (!(var->flags & VAR_FLAG_PARAMETER)) continue;
        scale(var->grad, gradient_scale);
        sub(var->val, var->val, var->grad);
    }

    free(returns);
    if (!save_policy(checkpoint_path, model)) return false;
    printf("LEARNED %u\n", sample_count);
    return true;
}

static int run_agent(model_state* model, SnakeENV* env, mem_arena* arena, u32 epochs, b32 force_retrain, b32 use_replay) {
    const char* checkpoint_path = getenv("SNAKE_POLICY_PATH");
    if (!checkpoint_path || !checkpoint_path[0]) checkpoint_path = "Snake-policy.bin";
    if (force_retrain || !load_policy(checkpoint_path, model)) {
        train(model, env, arena, epochs);
        if (!save_policy(checkpoint_path, model)) fprintf(stderr, "Could not save policy checkpoint.\n");
    } else {
        fprintf(stderr, "Loaded policy checkpoint.\n");
    }

    model_state inference_model = { 0 };
    create_actor_model(arena, &inference_model, 1);
    inference_model.forward_graph = build_graph(arena, &inference_model, inference_model.output);
    inference_model.cost_graph = build_graph(arena, &inference_model, inference_model.cost);
    if (!inference_model.forward_graph.vars || !inference_model.cost_graph.vars ||
        !copy_policy_parameters(model, &inference_model)) {
        fprintf(stderr, "Failed to create inference model.\n");
        return 1;
    }
    model = &inference_model;

    ReplayBuffer replay = { 0 };
    char* replay_path = NULL;
    if (use_replay) {
        size_t path_length = strlen(checkpoint_path);
        if (path_length > SIZE_MAX - sizeof(".replay.bin")) {
            fprintf(stderr, "Experience replay path is too long.\n");
            return 1;
        }
        replay_path = malloc(path_length + sizeof(".replay.bin"));
        replay.samples = malloc(sizeof(*replay.samples) * REPLAY_CAPACITY);
        if (!replay_path || !replay.samples) {
            fprintf(stderr, "Failed to allocate experience replay.\n");
            free(replay_path);
            free(replay.samples);
            return 1;
        }
        snprintf(replay_path, path_length + sizeof(".replay.bin"), "%s.replay.bin", checkpoint_path);
        if (force_retrain) {
            if (!save_replay(replay_path, &replay)) {
                fprintf(stderr, "Could not reset experience replay.\n");
                free(replay_path);
                free(replay.samples);
                return 1;
            }
        } else if (!load_replay(replay_path, &replay)) {
            free(replay_path);
            free(replay.samples);
            return 1;
        }
    }
    puts("READY");

    char line[8192];
    OnlineSample* samples = NULL;
    u32 sample_count = 0, sample_capacity = 0;
    f32 pending_observation[OBSERVATION_SIZE];
    ACTION pending_action = NONE;
    b32 has_pending_action = false;
    int result = 0;
    while (fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (strcmp(line, "ABORT") == 0) {
            sample_count = 0;
            has_pending_action = false;
            continue;
        }
        if (strcmp(line, "END") == 0) {
            if (has_pending_action) {
                puts("ERROR cannot end episode before feedback");
                continue;
            }
            if (!update_online_policy(model, samples, sample_count, checkpoint_path,
                                      use_replay ? &replay : NULL, replay_path)) {
                puts("ERROR online policy update or checkpoint save failed");
                result = 1;
                break;
            }
            sample_count = 0;
            continue;
        }
        if (strncmp(line, "FEEDBACK ", 9) == 0) {
            f32 reward = 0.0f;
            unsigned int done = 0;
            char extra;
            if (sscanf(line + 9, "%f %u %c", &reward, &done, &extra) != 2 ||
                !isfinite(reward) || done > 1 || !has_pending_action) {
                puts("ERROR invalid or unexpected episode feedback");
                continue;
            }
            if (!append_online_sample(&samples, &sample_count, &sample_capacity,
                                      pending_observation, pending_action, reward)) {
                puts("ERROR could not allocate online learning samples");
                result = 1;
                break;
            }
            has_pending_action = false;
            if (done) {
                if (!update_online_policy(model, samples, sample_count, checkpoint_path,
                                          use_replay ? &replay : NULL, replay_path)) {
                    puts("ERROR online policy update or checkpoint save failed");
                    result = 1;
                    break;
                }
                sample_count = 0;
            }
            continue;
        }

        f32 observation[OBSERVATION_SIZE];
        char* cursor = line;
        if (strncmp(cursor, "STEP ", 5) == 0) cursor += 5;
        b32 valid = true;
        for (u32 i = 0; i < OBSERVATION_SIZE; i++) {
            char* end = NULL;
            observation[i] = strtof(cursor, &end);
            if (end == cursor) {
                valid = false;
                break;
            }
            cursor = end;
        }
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n') cursor++;
        if (*cursor != '\0' || has_pending_action) valid = false;
        if (!valid) {
            puts("ERROR invalid observation or missing feedback");
            continue;
        }

        ACTION pov = RIGHT;
        f32 direction_max = -1.0f;
        for (u32 direction = 0; direction < 4; direction++) {
            f32 value = observation[GRID_SIZE + 4 + direction];
            if (value > direction_max) {
                direction_max = value;
                pov = (ACTION)direction;
            }
        }
        State head = {
            .x = (i32)(observation[GRID_SIZE] * (env->cols - 1) + 0.5f),
            .y = (i32)(observation[GRID_SIZE + 1] * (env->rows - 1) + 0.5f),
        };
        State food = {
            .x = (i32)(observation[GRID_SIZE + 2] * (env->cols - 1) + 0.5f),
            .y = (i32)(observation[GRID_SIZE + 3] * (env->rows - 1) + 0.5f),
        };
        State tail = {
            .x = (i32)(observation[GRID_SIZE + 8] * (env->cols - 1) + 0.5f),
            .y = (i32)(observation[GRID_SIZE + 9] * (env->rows - 1) + 0.5f),
        };
        State body[MAX_SNAKE_LENGTH];
        u32 body_len = 0;
        for (u32 cell = 0; cell < GRID_SIZE; cell++) {
            if (observation[cell] > 0.5f) {
                body[body_len++] = (State){ .x = (i32)(cell % env->cols), .y = (i32)(cell / env->cols) };
            }
        }
        for (u32 row = 0; row < OBSERVATION_SIZE; row++) {
            model->input->val->data[(u64)row * model->input->val->cols] = observation[row];
        }
        forward_pass(&model->forward_graph);
        pending_action = best_action_col(model->output->val, 0, pov, head, food, tail, body, body_len, env->cols, env->rows);
        memcpy(pending_observation, observation, sizeof(pending_observation));
        has_pending_action = true;
        printf("ACTION %u\n", (u32)pending_action);
    }
    free(samples);
    free(replay_path);
    free(replay.samples);
    return result;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);

    b32 agent_mode = argc > 1 && strcmp(argv[1], "--agent") == 0;
    if (argc > 1 && !agent_mode) {
        fprintf(stderr, "Usage: %s [--agent [epochs] [--retrain] [--replay]]\n", argv[0]);
        return 2;
    }

    u32 epochs = DEFAULT_TRAINING_EPOCHS;
    b32 force_retrain = false;
    b32 use_replay = false;
    b32 has_epoch_count = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--retrain") == 0 && !force_retrain) {
            force_retrain = true;
        } else if (strcmp(argv[i], "--replay") == 0 && !use_replay) {
            use_replay = true;
        } else if (!has_epoch_count) {
            char* end = NULL;
            unsigned long parsed = strtoul(argv[i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed > 1000000) {
                fprintf(stderr, "Invalid agent arguments.\n");
                return 2;
            }
            epochs = (u32)parsed;
            has_epoch_count = true;
        } else {
            fprintf(stderr, "Invalid agent arguments.\n");
            return 2;
        }
    }

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

    SnakeENV* env = create_env(GRID_SIZE);
    if (!env) {
        fprintf(stderr, "Failed to create environment!\n");
        arena_destroy(arena);
        return 1;
    }

    int result = 0;
    if (agent_mode) {
        result = run_agent(model, env, arena, epochs, force_retrain, use_replay);
    } else {
        printf("Starting Training Loop...\n");
        train(model, env, arena, epochs);
    }

    free(env);
    arena_destroy(arena);
    return result;
}
