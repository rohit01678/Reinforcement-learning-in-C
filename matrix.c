#include "matrix.h"
#include "random.h"

#include <math.h>
#include <string.h>

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

// NOTE on loop order: reordered to (i, k, j) instead of (i, j, k).
// This walks b->data and out->data contiguously in the innermost loop
// (stride 1) instead of striding through b by b->cols on every step.
// That's the difference between "cache-friendly" and "cache-hostile"
// access, and it lets the compiler auto-vectorize the inner loop.
// This is the dominant matmul case: every forward-pass layer
// (var_matmul_forward) always calls this exact case.
void _mat_mul_nn(matrix* restrict out, const matrix* restrict a, const matrix* restrict b) {
    for (u64 i = 0; i < out->rows; i++) {
        f32* restrict out_row = out->data + i * out->cols;
        const f32* restrict a_row = a->data + i * a->cols;
        for (u64 k = 0; k < a->cols; k++) {
            f32 aik = a_row[k];
            const f32* restrict b_row = b->data + k * b->cols;
            for (u64 j = 0; j < out->cols; j++) {
                out_row[j] += aik * b_row[j];
            }
        }
    }
}

// Dot product with 8 independent accumulators. The compiler can map this
// onto SIMD lanes without -ffast-math because the summation order is
// explicit in the source.
static inline f32 dot8(const f32* restrict x, const f32* restrict y, u64 n) {
    f32 s[8] = { 0 };
    u64 k = 0;
    for (; k + 8 <= n; k += 8)
        for (int l = 0; l < 8; l++) s[l] += x[k + l] * y[k + l];
    f32 tail = 0.0f;
    for (; k < n; k++) tail += x[k] * y[k];
    return ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7])) + tail;
}

// out += a * b^T : both operands are walked along contiguous rows.
void _mat_mul_nt(matrix* restrict out, const matrix* restrict a, const matrix* restrict b) {
    for (u64 i = 0; i < out->rows; i++) {
        const f32* a_row = a->data + i * a->cols;
        f32* out_row = out->data + i * out->cols;
        for (u64 j = 0; j < out->cols; j++)
            out_row[j] += dot8(a_row, b->data + j * b->cols, a->cols);
    }
}

// out += a^T * b : (i,k,j) order so the inner loop is stride-1 over b and out.
void _mat_mul_tn(matrix* restrict out, const matrix* restrict a, const matrix* restrict b) {
    for (u64 i = 0; i < out->rows; i++) {
        f32* restrict out_row = out->data + i * out->cols;
        for (u64 k = 0; k < a->rows; k++) {
            f32 aki = a->data[i + k * a->cols];
            const f32* restrict b_row = b->data + k * b->cols;
            for (u64 j = 0; j < out->cols; j++)
                out_row[j] += aki * b_row[j];
        }
    }
}

void _mat_mul_tt(matrix* restrict out, const matrix* restrict a, const matrix* restrict b) {
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

// Per-column softmax: each column of `in` is an independent probability
// distribution (batch dimension = columns).
b32 softmax(matrix* out, const matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    for (u32 j = 0; j < in->cols; j++) {
        f32 max_val = in->data[j];
        for (u32 i = 1; i < in->rows; i++)
            max_val = MAX(max_val, in->data[j + i * in->cols]);

        f32 sum = 0.0f;
        for (u32 i = 0; i < in->rows; i++) {
            f32 e = expf(in->data[j + i * in->cols] - max_val);
            out->data[j + i * in->cols] = e;
            sum += e;
        }
        for (u32 i = 0; i < in->rows; i++)
            out->data[j + i * in->cols] /= sum;
    }
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

    for (u32 j = 0; j < softmax_out->cols; j++) {
        f32 dot = 0.0f;
        for (u32 i = 0; i < softmax_out->rows; i++)
            dot += grad->data[j + i * grad->cols] * softmax_out->data[j + i * softmax_out->cols];

        for (u32 i = 0; i < softmax_out->rows; i++) {
            u32 idx = j + i * softmax_out->cols;
            out->data[idx] += softmax_out->data[idx] * (grad->data[idx] - dot);
        }
    }
    return true;
}

// ============================================================================
