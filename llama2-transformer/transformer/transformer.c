/* Transformer model implementation for CARTS benchmark */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 64
#define HIDDEN_DIM 256
#define N_LAYERS 2
#define N_HEADS 4
#define N_KV_HEADS 4
#define VOCAB_SIZE 256
#define SEQ_LEN 32

#define KV_DIM ((DIM * N_KV_HEADS) / N_HEADS)
#define KV_MUL (N_HEADS / N_KV_HEADS)
#define HEAD_SIZE (DIM / N_HEADS)

typedef struct {
  float token_embedding_table[VOCAB_SIZE * DIM];
  float rms_att_weight[N_LAYERS * DIM];
  float rms_ffn_weight[N_LAYERS * DIM];
  float wq[N_LAYERS * DIM * DIM];
  float wk[N_LAYERS * DIM * KV_DIM];
  float wv[N_LAYERS * DIM * KV_DIM];
  float wo[N_LAYERS * DIM * DIM];
  float w1[N_LAYERS * DIM * HIDDEN_DIM];
  float w2[N_LAYERS * HIDDEN_DIM * DIM];
  float w3[N_LAYERS * DIM * HIDDEN_DIM];
  float rms_final_weight[DIM];
} TransformerModel;

typedef struct {
  float x[DIM];
  float xb[DIM];
  float xb2[DIM];
  float hb[HIDDEN_DIM];
  float hb2[HIDDEN_DIM];
  float q_buf[DIM];
  float att_buf[N_HEADS * SEQ_LEN];
  float logits[VOCAB_SIZE];
  float key_cache[N_LAYERS * SEQ_LEN * KV_DIM];
  float value_cache[N_LAYERS * SEQ_LEN * KV_DIM];
} TransformerState;

static void zero_floats(float *buffer, int count);
static void rmsnorm(float *o, float *x_vec, float *weight, int size);
static void softmax(float *x_vec, int size);
static void matmul(float *xout, float *x_vec, float *w, int n, int d);
static void initialize_state(TransformerState *state);
static void initialize_test_data(TransformerModel *model);
static float *forward(const TransformerModel *model, TransformerState *state,
                      int token, int pos);

static void zero_floats(float *buffer, int count) {
  for (int i = 0; i < count; i++) {
    buffer[i] = 0.0f;
  }
}

static void rmsnorm(float *o, float *x_vec, float *weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) {
    ss += x_vec[j] * x_vec[j];
  }
  ss /= size;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (int j = 0; j < size; j++) {
    o[j] = weight[j] * (ss * x_vec[j]);
  }
}

static void softmax(float *x_vec, int size) {
  float max_val = x_vec[0];
  for (int i = 1; i < size; i++) {
    if (x_vec[i] > max_val) {
      max_val = x_vec[i];
    }
  }
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    x_vec[i] = expf(x_vec[i] - max_val);
    sum += x_vec[i];
  }
  for (int i = 0; i < size; i++) {
    x_vec[i] /= sum;
  }
}

static void matmul(float *xout, float *x_vec, float *w, int n, int d) {
  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x_vec[j];
    }
    xout[i] = val;
  }
}

static void initialize_state(TransformerState *state) {
  zero_floats(state->x, DIM);
  zero_floats(state->xb, DIM);
  zero_floats(state->xb2, DIM);
  zero_floats(state->hb, HIDDEN_DIM);
  zero_floats(state->hb2, HIDDEN_DIM);
  zero_floats(state->q_buf, DIM);
  zero_floats(state->att_buf, N_HEADS * SEQ_LEN);
  zero_floats(state->logits, VOCAB_SIZE);
  zero_floats(state->key_cache, N_LAYERS * SEQ_LEN * KV_DIM);
  zero_floats(state->value_cache, N_LAYERS * SEQ_LEN * KV_DIM);
}

static void initialize_test_data(TransformerModel *model) {
  for (int i = 0; i < VOCAB_SIZE * DIM; i++) {
    model->token_embedding_table[i] =
        0.01f * (float)(rand() % 100 - 50) / 50.0f;
  }

  for (int l = 0; l < N_LAYERS; l++) {
    for (int i = 0; i < DIM; i++) {
      model->rms_att_weight[l * DIM + i] = 1.0f;
      model->rms_ffn_weight[l * DIM + i] = 1.0f;
    }

    for (int i = 0; i < DIM * DIM; i++) {
      model->wq[l * DIM * DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      model->wo[l * DIM * DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }

    for (int i = 0; i < DIM * KV_DIM; i++) {
      model->wk[l * DIM * KV_DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      model->wv[l * DIM * KV_DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }

    for (int i = 0; i < DIM * HIDDEN_DIM; i++) {
      model->w1[l * DIM * HIDDEN_DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
      model->w3[l * DIM * HIDDEN_DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }

    for (int i = 0; i < HIDDEN_DIM * DIM; i++) {
      model->w2[l * HIDDEN_DIM * DIM + i] = 0.01f * (float)(rand() % 100 - 50) / 50.0f;
    }
  }

  for (int i = 0; i < DIM; i++) {
    model->rms_final_weight[i] = 1.0f;
  }
}

// -------------------------------------------------------------------
// Transformer forward pass
static float *forward(const TransformerModel *model, TransformerState *state,
                      int token, int pos) {
  // a few convenience variables
  float *x_vec = state->x;
  float *xb_vec = state->xb;
  float *xb2_vec = state->xb2;
  float *hb_vec = state->hb;
  float *hb2_vec = state->hb2;
  float *q_vec = state->q_buf;
  float *att = state->att_buf;

  // copy the token embedding into x
  int content_base = token * DIM;
  for (int i = 0; i < DIM; i++) {
    x_vec[i] = model->token_embedding_table[content_base + i];
  }

  // forward all the layers
  for (int l = 0; l < N_LAYERS; l++) {
    // attention rmsnorm
    rmsnorm(xb_vec, x_vec, model->rms_att_weight + l * DIM, DIM);

    // key and value point to the kv cache
    int layer_offset = l * SEQ_LEN * KV_DIM;
    int current_offset = layer_offset + pos * KV_DIM;

    // qkv matmuls for this position
    matmul(q_vec, xb_vec, model->wq + l * DIM * DIM, DIM, DIM);
    matmul(&state->key_cache[current_offset], xb_vec,
           model->wk + l * DIM * KV_DIM, DIM, KV_DIM);
    matmul(&state->value_cache[current_offset], xb_vec,
           model->wv + l * DIM * KV_DIM, DIM, KV_DIM);

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (int i = 0; i < DIM; i += 2) {
      int head_dim = i % HEAD_SIZE;
      float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)HEAD_SIZE);
      float angle = pos * freq;
      float fcr = cosf(angle);
      float fci = sinf(angle);
      int rotn = i < KV_DIM ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (int v_idx = 0; v_idx < rotn; v_idx++) {
        float *vec =
            (v_idx == 0) ? q_vec : &state->key_cache[current_offset];
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    int h;
#pragma omp parallel for private(h)
    for (h = 0; h < N_HEADS; h++) {
      // get the query vector for this head
      float *head_q = q_vec + h * HEAD_SIZE;
      // attention scores for this head
      float *head_att = att + h * SEQ_LEN;
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        int head_k_offset = layer_offset + t * KV_DIM + ((h / KV_MUL) * HEAD_SIZE);
        float *head_k = &state->key_cache[head_k_offset];
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < HEAD_SIZE; i++) {
          score += head_q[i] * head_k[i];
        }
        score /= sqrtf((float)HEAD_SIZE);
        // save the score to the attention buffer
        head_att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      softmax(head_att, pos + 1);

      // weighted sum of the values, store back into xb
      float *xb_head = xb_vec + h * HEAD_SIZE;
      zero_floats(xb_head, HEAD_SIZE);
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        int head_v_offset =
            layer_offset + t * KV_DIM + (h / KV_MUL) * HEAD_SIZE;
        float *head_v = &state->value_cache[head_v_offset];
        // get the attention weight for this timestep
        float weight = head_att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < HEAD_SIZE; i++) {
          xb_head[i] += weight * head_v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    matmul(xb2_vec, xb_vec, model->wo + l * DIM * DIM, DIM, DIM);

    // residual connection back into x
    for (int i = 0; i < DIM; i++) {
      x_vec[i] += xb2_vec[i];
    }

    // ffn rmsnorm
    rmsnorm(xb_vec, x_vec, model->rms_ffn_weight + l * DIM, DIM);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(hb_vec, xb_vec, model->w1 + l * DIM * HIDDEN_DIM, DIM, HIDDEN_DIM);
    matmul(hb2_vec, xb_vec, model->w3 + l * DIM * HIDDEN_DIM, DIM, HIDDEN_DIM);

    // SwiGLU non-linearity
    for (int i = 0; i < HIDDEN_DIM; i++) {
      float val = hb_vec[i];
      // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
      val *= (1.0f / (1.0f + expf(-val)));
      // elementwise multiply with w3(x)
      val *= hb2_vec[i];
      hb_vec[i] = val;
    }

    // final matmul to get the output of the ffn
    matmul(xb_vec, hb_vec, model->w2 + l * HIDDEN_DIM * DIM, HIDDEN_DIM, DIM);

    // residual connection
    for (int i = 0; i < DIM; i++) {
      x_vec[i] += xb_vec[i];
    }
  }

  // final rmsnorm
  rmsnorm(x_vec, x_vec, model->rms_final_weight, DIM);

  // classifier into logits
  matmul(state->logits, x_vec, model->token_embedding_table, DIM, VOCAB_SIZE);
  return state->logits;
}

int main(void) {
  printf("Testing isolated Transformer neural network functions\n");
  printf("Configuration: dim=%d, hidden_dim=%d, n_layers=%d, n_heads=%d, "
         "vocab_size=%d\n",
         DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, VOCAB_SIZE);

  TransformerModel model;
  TransformerState state;

  initialize_state(&state);
  srand(42);
  initialize_test_data(&model);

  printf("Testing forward pass...\n");
  int test_token = 42;
  int test_pos = 0;
  float *logits_out = forward(&model, &state, test_token, test_pos);

  printf("Forward pass completed. First 10 logits: ");
  for (int i = 0; i < 10; i++) {
    printf("%.4f ", logits_out[i]);
  }
  printf("\n");

  printf("\nTesting individual functions...\n");

  float test_x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float test_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  float test_o[4];
  rmsnorm(test_o, test_x, test_weight, 4);
  printf("RMSNorm test: [%.4f, %.4f, %.4f, %.4f] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_x[0], test_x[1], test_x[2], test_x[3], test_o[0], test_o[1],
         test_o[2], test_o[3]);

  float test_softmax[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  softmax(test_softmax, 4);
  printf("Softmax test: [1.0, 2.0, 3.0, 4.0] -> [%.4f, %.4f, %.4f, %.4f]\n",
         test_softmax[0], test_softmax[1], test_softmax[2], test_softmax[3]);

  float test_w[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  float test_vec[3] = {1.0f, 1.0f, 1.0f};
  float test_result[2];
  matmul(test_result, test_vec, test_w, 3, 2);
  printf("Matmul test: [1,2,3; 4,5,6] @ [1,1,1] = [%.1f, %.1f]\n",
         test_result[0], test_result[1]);

  printf("All tests completed successfully!\n");
  return 0;
}
