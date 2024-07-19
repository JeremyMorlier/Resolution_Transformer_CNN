import torch
from torchinfo import summary


# from https://www.adamcasson.com/posts/transformer-flops#user-content-fn-dm_scale
def flops_per_sequence(patch_size, n_patch, n_layers, n_heads, d_model, d_mlp, n_classes):
    """DeepMind method for forwad pass FLOPs counting of decoder-only Transformer
    """
    # Conv2d Embedding
    sequence_length = n_patch + 1
    embedding_flops = 2*(n_patch)*patch_size*patch_size*3*d_model

    QKV_flops = 2*3*sequence_length*d_model*d_model
    QK_logits_flops = 2*sequence_length*sequence_length*d_model
    softmax_flops = 3*n_heads*sequence_length*sequence_length
    attention_reduction_flops = 2*sequence_length*sequence_length*d_model
    attention_project_flops = 2*sequence_length*d_model*d_model
    feedforward_flops = 4*sequence_length*d_model*d_mlp
 
    logits = 2*d_model*n_classes
    total_attn_flops = QKV_flops + QK_logits_flops + softmax_flops + attention_reduction_flops + attention_project_flops + feedforward_flops

    return embedding_flops + n_layers * (total_attn_flops + feedforward_flops) + logits, embedding_flops, total_attn_flops, feedforward_flops, logits

def get_memory_flops(model, resolution, args) :

    memory = 0.0
    flops = 0.0

    if "resnet" in args.model or "regseg" in args.model:
        info = summary(model, (1, 3, resolution, resolution), verbose=0, col_names=("output_size", "num_params", "mult_adds"))
        memory = info.max_memory
        flops = 2 * info.total_mult_adds
    elif "vit" in args.model :
        patch_number = int(resolution * resolution / (args.patch_size * args.patch_size))
        flops = list(flops_per_sequence(args.patch_size, patch_number, args.num_layers, args.num_heads, args.hidden_dim, args.mlp_dim, 1000))

    return memory, flops
