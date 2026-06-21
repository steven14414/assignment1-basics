V = 50257
T = 1024


def nearest_multiple_64(x):
    lo = int(x // 64) * 64
    hi = lo + 64
    return lo if abs(x - lo) <= abs(hi - x) else hi


def flops(L, D, H, T, V=50257, d_ff=None):
    if d_ff is None:
        d_ff = nearest_multiple_64(8 / 3 * D)
    params_per_layer = 4 * D * D + 3 * D * d_ff + 2 * D
    params = 2 * V * D + L * params_per_layer + D
    # per layer matrix multiply FLOPs
    qkv = 2 * T * D * (3 * D)
    attn_scores = 2 * H * T * T * (D // H)  # QK^T
    attn_values = 2 * H * T * T * (D // H)  # softmax(QK)V
    o = 2 * T * D * D
    ffn = 2 * T * (D * d_ff + D * d_ff + d_ff * D)
    per_layer = qkv + attn_scores + attn_values + o + ffn
    all_layers = L * per_layer
    lm_head = 2 * T * D * V
    total = all_layers + lm_head
    return dict(
        L=L,
        D=D,
        H=H,
        T=T,
        d_ff=d_ff,
        params=params,
        params_gb=params * 4 / 1e9,
        params_gib=params * 4 / 1024**3,
        qkv=qkv,
        attn_scores=attn_scores,
        attn_values=attn_values,
        o=o,
        ffn=ffn,
        per_layer=per_layer,
        all_layers=all_layers,
        lm_head=lm_head,
        total=total,
        attn_proj_layers=L * (qkv + o),
        attn_quad_layers=L * (attn_scores + attn_values),
        ffn_layers=L * ffn,
    )


def sci(x):
    return f"{x:.6e}"


def human_count(x):
    if abs(x) >= 1e9:
        return f"{x / 1e9:.3f}B"
    if abs(x) >= 1e6:
        return f"{x / 1e6:.3f}M"
    return f"{x:.0f}"


def print_breakdown(name, result):
    print(f"===={name}====")
    print(f"d_ff: {result['d_ff']}")
    print(f"params: {human_count(result['params'])}")
    print(f"param_memory_gb: {result['params_gb']:.3f}GB")
    print(f"param_memory_gib: {result['params_gib']:.3f}GiB")
    print(f"per_layer_total: {sci(result['per_layer'])}")
    print(f"layers_total: {sci(result['all_layers'])}")
    print(f"lm_head: {sci(result['lm_head'])}")
    print(f"total: {sci(result['total'])}")
    for comp in ["attn_proj_layers", "attn_quad_layers", "ffn_layers", "lm_head"]:
        print(f"{comp}: {sci(result[comp])} ({result[comp] / result['total']:.2%})")
    print()


models = {
    "small": (12, 768, 12),
    "medium": (24, 1024, 16),
    "large": (36, 1280, 20),
    "XL": (48, 1600, 25),
}
for name, (L, D, H) in models.items():
    r = flops(L, D, H, 1024)
    print_breakdown(f"{name} T=1024", r)

r = flops(48, 1600, 25, 16384, d_ff=4288)
print_breakdown("XL T=16384", r)
