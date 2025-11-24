################################################################################
# Problem-size presets for ML kernels. Examples automatically pick up the
# defaults and expose matching phony targets (mini/small/medium/standard/large).
################################################################################

# Activations - element-wise operations on flat arrays
DEFAULT_CFLAGS_activations := -DSIZE=1048576
PRESET_TARGETS_activations := mini small medium standard large
PRESET_FLAGS_activations_mini := -DSIZE=1024
PRESET_FLAGS_activations_small := -DSIZE=65536
PRESET_FLAGS_activations_medium := -DSIZE=1048576
PRESET_FLAGS_activations_standard := -DSIZE=1048576
PRESET_FLAGS_activations_large := -DSIZE=16777216

# Batch normalization - 4D tensors (batch, channels, height, width)
DEFAULT_CFLAGS_batchnorm := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32
PRESET_TARGETS_batchnorm := mini small medium standard large
PRESET_FLAGS_batchnorm_mini := -DBATCH_SIZE=2 -DCHANNELS=8 -DHEIGHT=8 -DWIDTH=8
PRESET_FLAGS_batchnorm_small := -DBATCH_SIZE=4 -DCHANNELS=16 -DHEIGHT=16 -DWIDTH=16
PRESET_FLAGS_batchnorm_medium := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32
PRESET_FLAGS_batchnorm_standard := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=32 -DWIDTH=32
PRESET_FLAGS_batchnorm_large := -DBATCH_SIZE=8 -DCHANNELS=128 -DHEIGHT=64 -DWIDTH=64

# Pooling - 4D tensors with pooling window
DEFAULT_CFLAGS_pooling := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=64 -DWIDTH=64 -DPOOL_SIZE=2
PRESET_TARGETS_pooling := mini small medium standard large
PRESET_FLAGS_pooling_mini := -DBATCH_SIZE=1 -DCHANNELS=8 -DHEIGHT=16 -DWIDTH=16 -DPOOL_SIZE=2
PRESET_FLAGS_pooling_small := -DBATCH_SIZE=2 -DCHANNELS=16 -DHEIGHT=32 -DWIDTH=32 -DPOOL_SIZE=2
PRESET_FLAGS_pooling_medium := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=64 -DWIDTH=64 -DPOOL_SIZE=2
PRESET_FLAGS_pooling_standard := -DBATCH_SIZE=4 -DCHANNELS=64 -DHEIGHT=64 -DWIDTH=64 -DPOOL_SIZE=2
PRESET_FLAGS_pooling_large := -DBATCH_SIZE=8 -DCHANNELS=128 -DHEIGHT=128 -DWIDTH=128 -DPOOL_SIZE=3

# LayerNorm - 2D tensors (batch, hidden_dim)
DEFAULT_CFLAGS_layernorm := -DBATCH=16 -DHIDDEN=1024
PRESET_TARGETS_layernorm := mini small medium standard large
PRESET_FLAGS_layernorm_mini := -DBATCH=4 -DHIDDEN=256
PRESET_FLAGS_layernorm_small := -DBATCH=8 -DHIDDEN=512
PRESET_FLAGS_layernorm_medium := -DBATCH=16 -DHIDDEN=1024
PRESET_FLAGS_layernorm_standard := -DBATCH=16 -DHIDDEN=1024
PRESET_FLAGS_layernorm_large := -DBATCH=64 -DHIDDEN=4096
