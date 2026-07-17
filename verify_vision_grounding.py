import copy
import torch
import torch.nn as nn
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
assert out.shape == (B, rm_num_slots * d_model), f"Shape mismatch: {out.shape}"

# backward compatibility -- no visual input
out2 = rm.forward_step(torch.randn(B, d_model), rm.init_memory(B))
assert out2.shape == (B, rm_num_slots * d_model), f"Shape mismatch: {out2.shape}"

# full sequence forward
outputs = rm.forward(torch.randn(B, seq_len, d_model), rm.init_memory(B),
                      visual_feats=visual_feats, visual_mask=visual_mask)
assert outputs.shape == (B, seq_len, rm_num_slots * d_model), f"Shape mismatch: {outputs.shape}"

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
assert out.shape == (B, seq_len, d_model), f"Shape mismatch: {out.shape}"

print("ALL SMOKE TESTS PASSED.")
