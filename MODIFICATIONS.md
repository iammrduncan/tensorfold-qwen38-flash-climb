# Modifications

This repository is [TensorFold](https://github.com/ashhart/TensorFold), release v1.2.3 (commit `6ec5fa4cb233f3606acbdad5bec433f0f8758f96`), by Ash Hart and the TensorFold contributors, under its licences (`LICENSE`, `NOTICE`, `LICENSES/`), with the changes below. As the Apache License 2.0 asks, this file states which files were changed.

## The recipe's patches, brought to v1.2.3

Three patches of MiaAI-Lab's recipe [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (`patches/` at `20e15e41f2f514a0d287710f4665ee1b5009ba41`), by MiaAI-Lab under the MIT License (`LICENSES/MiaAI-Lab-MIT.txt`, and the recipe's notice in `NOTICE-recipe`), each as its own commit: those the recipe needs to serve its way, with fp8 KV, vision and its image limit.

- `0001-vision-nvidia-fp4-tower` and `0002-vision-image-tokens-16384` apply to v1.2.3 as they are.
- `0003-kv-cache-fp8` was written for v1.1.4 and is carried over: its commit says what changed.

The recipe's other 14 patches are not here. The files these three change:

- `docs/recipes/flash-next-cuda.md`
- `docs/vision.md`
- `zig/kernels/cuda/flashnext_attn.cu`
- `zig/src/families/flashnext/cuda_batch_lanes.zig`
- `zig/src/families/flashnext/cuda_batch_state.zig`
- `zig/src/families/flashnext/cuda_batch.zig`
- `zig/src/families/flashnext/cuda_disk.zig`
- `zig/src/families/flashnext/cuda_glue.zig`
- `zig/src/families/flashnext/cuda_images.zig`
- `zig/src/families/flashnext/cuda_kv.zig`
- `zig/src/families/flashnext/cuda_lanes_explain.zig`
- `zig/src/families/flashnext/cuda_lanes_load.zig`
- `zig/src/families/flashnext/cuda_lanes_snap.zig`
- `zig/src/families/flashnext/cuda_lanes.zig`
- `zig/src/families/flashnext/cuda_launch.zig`
- `zig/src/families/flashnext/cuda_media.zig`
- `zig/src/families/flashnext/cuda_mtp_draft.zig`
- `zig/src/families/flashnext/cuda_mtp_ops.zig`
- `zig/src/families/flashnext/cuda_prompt.zig`
- `zig/src/families/flashnext/cuda_seq_bytes.zig`
- `zig/src/families/flashnext/cuda_seq_parts.zig`
- `zig/src/families/flashnext/cuda_snap.zig`
- `zig/src/native/cuda_flashnext_kv8_test.zig`
- `zig/src/native/cuda.zig`
- `zig/src/vision/qwen_cuda_dequant.zig`
- `zig/src/vision/qwen_cuda_media.zig`
- `zig/src/vision/qwen_cuda_weights_logs.zig`
- `zig/src/vision/qwen_cuda_weights_quant_test.zig`
- `zig/src/vision/qwen_cuda_weights_test.zig`
- `zig/src/vision/qwen_cuda_weights.zig`
- `zig/tests/cuda/flashnext_kv8_tests.zig`
- `zig/tests/cuda/main.zig`

## The first token's logprobs on Flash Next's CUDA engine

So the hill climb's judge can check what a change does to the model's output distribution, Flash Next's CUDA lanes give the first token's logprob row (OpenAI `logprobs`, with up to 20 best tokens and the rank): the prompt's last row of scores, the one the first token is drawn from, read in f64 on the host. A request with `logprobs` and more than one token is refused in words, since a window's rows are not read back. The files:

- `zig/src/families/flashnext/cuda_lanes.zig`
- `zig/src/families/flashnext/cuda_lanes_explain.zig`
- `zig/src/families/flashnext/cuda_logprob_row.zig` (new)
