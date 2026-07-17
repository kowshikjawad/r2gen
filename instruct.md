# Project Context: Vision-Grounded Recurrent Relational Memory

You are modifying the official PyTorch implementation of R2Gen (`cuhksz-nlp/R2Gen`).
The current architecture uses a text-driven recurrent relational memory update.
The goal of this task is to implement a "Vision-Grounded Recurrent Update" by
inserting a cross-attention step over the encoder's visual features *before* the
memory is updated with the target token embedding.

This is an ablation study. You must preserve the original gated memory logic exactly
as designed, only adding the visual grounding step.

## Target File

`modules/encoder_decoder.py` (specifically the `RelationalMemory` module and the
`Transformer` decoding logic).

## Implementation Requirements

### 1. Modify `RelationalMemory` Initialization

- In `__init__`, add a new multi-head attention module for the visual grounding step.
  The correct class name in this codebase is `MultiHeadedAttention` (with "-ed") —
  confirm this against the existing `self.attn = MultiHeadedAttention(num_heads, d_model)`
  line already present in the same `__init__`, and use the identical class:
  ```python
  self.visual_attn = MultiHeadedAttention(num_heads, d_model)
  ```
- Ensure it utilizes the existing `num_heads` and `d_model` parameters already passed
  into `RelationalMemory.__init__` — do not introduce new constructor arguments.

### 2. Modify `RelationalMemory.forward_step`

- Update the method signature to accept visual features **as keyword arguments with
  `None` defaults**, not required positional arguments. This keeps the method
  backward-compatible (calling it with no visual input reproduces the exact original
  R2Gen behavior), which matters for cleanly running the baseline arm of the ablation
  from the same codebase if needed:
  ```python
  def forward_step(self, input, memory, visual_feats=None, visual_mask=None):
  ```
- **Phase 1: Visual Grounding (NEW):**
  - Reshape the memory matrix as the existing code already does:
    `memory = memory.reshape(-1, self.num_slots, self.d_model)`
  - Run the memory through `self.visual_attn` acting as the Query, with `visual_feats`
    acting as both the Key and Value, guarded so this step is skipped entirely when
    no visual features are supplied:
    ```python
    if visual_feats is not None:
        memory = memory + self.visual_attn(memory, visual_feats, visual_feats, visual_mask)
    ```
  - `visual_mask` may be `None` (fine for IU X-Ray, which uses a fixed-length 49-patch
    grid with no padding). If this is later pointed at a dataset with variable-length
    visual sequences, pass the corresponding attention mask instead of leaving it `None`.
- **Phase 2: Gated Text Integration (UNCHANGED):**
  - Do not modify the existing R2Gen logic that follows the visual grounding step.
  - Note: this logic is **not** encapsulated in a separate method — there is no
    `self._gate` method in this codebase. The gating logic (the `q`/`k`/`v`
    assignments, `self.attn`, `self.mlp`, and the inline `self.W`/`self.U`
    sigmoid-gated combination) is written directly in the body of `forward_step`,
    immediately after the block above. Leave all of it completely intact, operating
    on the newly grounded `memory` state exactly as it currently operates on `memory`.

### 3. Modify `RelationalMemory.forward`

This method is not mentioned as a separate step in earlier drafts of this task, but
**it must be updated or the change will not compile.** `forward` is the loop that
calls `forward_step` once per decoding timestep — its signature and call site must be
extended to match the new `forward_step` signature from Step 2, or the call will bind
arguments incorrectly and raise a `TypeError` the first time the model runs:
```python
def forward(self, inputs, memory, visual_feats=None, visual_mask=None):
    outputs = []
    for i in range(inputs.shape[1]):
        memory = self.forward_step(inputs[:, i], memory, visual_feats=visual_feats, visual_mask=visual_mask)
        outputs.append(memory)
    outputs = torch.stack(outputs, dim=1)
    return outputs
```

### 4. Modify `Transformer.decode`

- The visual features (`hidden_states` out of the encoder) must be threaded into the
  memory update at each decoding step, along with the corresponding mask.
- Locate the line where the relational memory is updated:
  - *Original:*
    ```python
    memory = self.rm(self.tgt_embed(tgt), memory)
    ```
  - *Target:*
    ```python
    memory = self.rm(self.tgt_embed(tgt), memory, visual_feats=hidden_states, visual_mask=src_mask)
    ```
  - Use keyword arguments here (`visual_feats=`, `visual_mask=`) rather than
    positional, to match the signature from Step 3 and avoid ambiguity with the
    existing `memory` positional argument.
- `core()` (used during beam search / greedy decoding elsewhere in this file) already
  passes the visual features and mask into `self.model.decode(...)` under the local
  variable names `memory` and `mask` — no changes to `core()` are needed for this to
  work at inference time.

## Dimensionality Constraints

- The visual features are already projected to `d_model` (e.g., 512) before reaching
  the decoder. This projection happens in `EncoderDecoder._prepare_feature_forward`
  in `modules/att_model.py`, via `self.att_embed` (a `Linear` layer) — **not** in
  `Transformer.forward`, which only calls `encode()` then `decode()` and performs no
  projection itself. Verify the projection there if needed; do not add a new
  projection layer inside `RelationalMemory`.
- Ensure tensor shapes align during the `visual_attn` call: `memory` is
  `(batch_size, num_slots, d_model)` after the reshape above, and `visual_feats` is
  `(batch_size, num_patches, d_model)` (49 patches for a single IU X-Ray view). These
  shapes are compatible with `MultiHeadedAttention` as a standard query/key/value
  cross-attention call with no further reshaping required.

## Verification (required before considering this task complete)

Do not rely on visual inspection alone — run a smoke test with fake tensors shaped
like the real pipeline before this is considered done, since a shape or signature
mismatch here would otherwise only surface after a full training run:

```python
import copy, torch, torch.nn as nn
from modules.encoder_decoder import RelationalMemory, Transformer, Encoder, EncoderLayer, \
    Decoder, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward, \
    PositionalEncoding, Embeddings

B, S_img, d_model, num_heads, d_ff, num_layers = 2, 49, 512, 8, 512, 3
rm_num_slots, rm_num_heads, vocab, seq_len = 3, 8, 100, 10

rm = RelationalMemory(num_slots=rm_num_slots, d_model=d_model, num_heads=rm_num_heads)
memory = rm.init_memory(B)
visual_feats = torch.randn(B, S_img, d_model)
visual_mask = torch.ones(B, 1, S_img)

# forward_step with visual grounding
out = rm.forward_step(torch.randn(B, d_model), memory, visual_feats=visual_feats, visual_mask=visual_mask)
assert out.shape == (B, rm_num_slots * d_model)

# backward compatibility -- no visual input
out2 = rm.forward_step(torch.randn(B, d_model), rm.init_memory(B))
assert out2.shape == (B, rm_num_slots * d_model)

# full sequence forward
outputs = rm.forward(torch.randn(B, seq_len, d_model), rm.init_memory(B),
                      visual_feats=visual_feats, visual_mask=visual_mask)
assert outputs.shape == (B, seq_len, rm_num_slots * d_model)

# full Transformer.decode() end-to-end
c = copy.deepcopy
attn = MultiHeadedAttention(num_heads, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, 0.1)
position = PositionalEncoding(d_model, 0.1)
model = Transformer(
    Encoder(EncoderLayer(d_model, c(attn), c(ff), 0.1), num_layers),
    Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), 0.1, rm_num_slots, d_model), num_layers),
    lambda x: x,
    nn.Sequential(Embeddings(d_model, vocab), c(position)),
    RelationalMemory(num_slots=rm_num_slots, d_model=d_model, num_heads=rm_num_heads),
)
src_mask = torch.ones(B, 1, S_img)
tgt = torch.randint(1, vocab, (B, seq_len))
tgt_mask = torch.ones(seq_len, seq_len).tril().unsqueeze(0).expand(B, -1, -1).bool()
hidden_states = model.encode(torch.randn(B, S_img, d_model), src_mask)
out = model.decode(hidden_states, src_mask, tgt, tgt_mask)
assert out.shape == (B, seq_len, d_model)

print("ALL SMOKE TESTS PASSED.")
```

If any assertion fails, do not proceed to training — report the exact error and the
shapes involved.

## Output Expectations

Do not rewrite the entire file. Apply these specific surgical changes to
`modules/encoder_decoder.py` — Steps 1, 2, 3, and 4 above are all required; Step 3
(updating `forward`) is easy to overlook since it isn't where the new logic lives,
but skipping it will break the call chain. Ensure the code is clean, well-commented
with `# NEW: vision-grounding step`, keep `visual_feats`/`visual_mask` as optional
keyword arguments throughout, and confirm the smoke test above passes before treating
this as ready to run against the IU X-Ray dataset.