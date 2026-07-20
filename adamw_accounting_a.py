def calc_memory(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 8 / 3 * d_model
    parameters = 4 * (
        vocab_size * d_model
        + num_layers * (d_model + d_model * d_model * 4 + d_model + d_model * d_ff * 3)
        + d_model
        + vocab_size * d_model
    )
    gradients = parameters
    optimizer_state = parameters * 2
    activations = (
        4
        * batch_size
        * (
            num_layers
            * (
                context_length * (d_model + d_model + d_model * 3)
                + context_length * context_length * num_heads * 2
                + context_length * d_model * 2
                + context_length * d_ff * 5
            )
            + context_length * d_model
            + context_length * vocab_size * 2
        )
    )
    # fp32*(num_layers*transformer_block+final_RMSNorm+lm_head+cross_entropy)
    return parameters + gradients + optimizer_state + activations
