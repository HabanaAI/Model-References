from fairseq.utils import safe_getattr
from fairseq.models import register_model_architecture
from fairseq.models.transformer_lm import base_lm_architecture
from fairseq.model_parallel.models.transformer_lm import base_lm_architecture as base_lm_mp_architecture


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_gpt2_tiny"
)
def transformer_lm_mp_gpt2_tiny(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 64)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 64)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 2)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 1)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_gpt"
)
def transformer_lm_mp_gpt(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_gpt2_12b"
)
def transformer_lm_mp_gpt2_12b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 4096)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 4096 * 4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 60)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_meg_1p2b"
)
def transformer_lm_mp_meg_1p2b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 1536*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_meg_2p5b"
)
def transformer_lm_mp_meg_2p5b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1920)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 1920*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 54)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 20)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "model_parallel_transformer_lm", "transformer_lm_mp_meg_8p3b"
)
def transformer_lm_mp_meg_8p3b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 3072)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 72)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_mp_architecture(args)


@register_model_architecture(
    "transformer_lm", "transformer_lm_meg_1p2b"
)
def transformer_lm_meg_1p2b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 1536*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture(
    "transformer_lm", "transformer_lm_meg_2p5b"
)
def transformer_lm_meg_2p5b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1920)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 1920*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 54)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 20)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture(
    "transformer_lm", "transformer_lm_meg_8p3b"
)
def transformer_lm_meg_8p3b(args):
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 3072)
    args.decoder_ffn_embed_dim = safe_getattr(args, "decoder_ffn_embed_dim", 3072*4)
    args.decoder_layers = safe_getattr(args, "decoder_layers", 72)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)
