"""AdamW training resource accounting (CS336 assignment 1, problem adamw_accounting)."""

BYTES_PER_FLOAT32 = 4


def param_count(vocab_size, num_layers, d_model, d_ff):
    """Trainable parameter count for the assignment Transformer LM."""
    per_layer = 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model
    return 2 * vocab_size * d_model + num_layers * per_layer + d_model


def calc_memory_breakdown(
    batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff
):
    """Peak memory in bytes, decomposed into parameters / activations / gradients / optimizer state."""
    n_params = param_count(vocab_size, num_layers, d_model, d_ff)

    parameters = BYTES_PER_FLOAT32 * n_params
    gradients = parameters
    optimizer_state = 2 * parameters  # first and second moment estimates

    activations = BYTES_PER_FLOAT32 * batch_size * (
        num_layers
        * (
            context_length * (5 * d_model)  # 2x RMSNorm + QKV projection output
            + context_length * context_length * num_heads * 2  # QK^T scores + softmax-weighted values
            + context_length * d_model * 2  # attention output projection + residual stream
            + context_length * d_ff * 5  # W1, W2, SiLU, gate product, W3 (SwiGLU)
        )
        + context_length * d_model  # final RMSNorm
        + context_length * vocab_size * 2  # lm_head logits + cross-entropy intermediates
    )

    total = parameters + gradients + optimizer_state + activations
    return {
        "parameters": parameters,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "activations": activations,
        "total": total,
        "n_params": n_params,
    }


def calc_memory(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 8 / 3 * d_model
    return calc_memory_breakdown(
        batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff
    )["total"]


def forward_flops(batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """Forward-pass FLOPs for one batch (matrix multiplies only)."""
    qkv = 2 * context_length * d_model * (3 * d_model)
    attn_scores = 2 * num_heads * context_length**2 * (d_model // num_heads)
    attn_values = 2 * num_heads * context_length**2 * (d_model // num_heads)
    output_proj = 2 * context_length * d_model * d_model
    ffn = 2 * context_length * (d_model * d_ff + d_model * d_ff + d_ff * d_model)
    per_layer = qkv + attn_scores + attn_values + output_proj + ffn
    lm_head = 2 * context_length * d_model * vocab_size
    return batch_size * (num_layers * per_layer + lm_head)


def training_step_flops(batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """One optimizer step: forward + backward (2x forward) + negligible AdamW elementwise ops."""
    return 3 * forward_flops(
        batch_size, vocab_size, context_length, num_layers, d_model, num_heads, d_ff
    )


def gpt2_xl_config():
    return dict(
        vocab_size=50257,
        context_length=1024,
        num_layers=48,
        d_model=1600,
        num_heads=25,
        d_ff=4288,
    )


if __name__ == "__main__":
    cfg = gpt2_xl_config()
    static = calc_memory_breakdown(0, **cfg)
    per_batch = calc_memory_breakdown(1, **cfg)["activations"]
    static_bytes = static["parameters"] + static["gradients"] + static["optimizer_state"]

    print("=== (a) symbolic breakdown (bytes, d_ff = 8/3 * d_model) ===")
    print("N_p = 2VD + L(4D^2 + 3D d_ff + 2D) + D")
    print("parameters   = 4 N_p")
    print("gradients    = 4 N_p")
    print("optimizer    = 8 N_p")
    print(
        "activations  = 4B [ L T (5D + 2TH + 2D + 5 d_ff) + TD + 2TV ]"
    )
    print("total        = 16 N_p + activations")

    print("\n=== (b) GPT-2 XL, d_ff=4288 ===")
    print(f"static term b = {static_bytes / 1e9:.6f} GB")
    print(f"batch coeff a = {per_batch / 1e9:.6f} GB / batch")
    max_batch = int((80e9 - static_bytes) // per_batch)
    print(f"max batch size within 80 GB = {max_batch}")

    print("\n=== (c) one training step FLOPs ===")
    fwd = forward_flops(1, **cfg)
    step = training_step_flops(1, **cfg)
    print(f"forward (B=1):  {fwd:.6e}")
    print(f"one step (B=1): {step:.6e}  (= 3 x forward)")

    print("\n=== (d) GPT-2 XL, B=1024, 400K steps, 50% MFU on H100 ===")
    batch_size = 1024
    steps = 400_000
    total_flops = training_step_flops(batch_size, **cfg) * steps
    effective_tflops = 495 * 0.5
    hours = total_flops / (effective_tflops * 1e12) / 3600
    print(f"total FLOPs: {total_flops:.6e}")
    print(f"training time: {hours:.1f} hours ({hours / 24:.1f} days)")
