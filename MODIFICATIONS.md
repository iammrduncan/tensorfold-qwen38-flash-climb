# Modifications

This repository is [TensorFold](https://github.com/ashhart/TensorFold), release v1.1.4 (commit `eab8582819688325e371907e4706e1c602b8a273`), by Ash Hart and the TensorFold contributors, under its licences (`LICENSE`, `NOTICE`, `LICENSES/`), with the changes below. As the Apache License 2.0 asks, this file states which files were changed.

## The recipe's patches

The 17 patches of MiaAI-Lab's recipe [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (`patches/` at `20e15e41f2f514a0d287710f4665ee1b5009ba41`), by MiaAI-Lab under the MIT License (`LICENSES/MiaAI-Lab-MIT.txt`, and the recipe's notice in `NOTICE-recipe`), each applied as its own commit, as the recipe's `scripts/prepare.sh` applies it. The files they change:

- `CHANGELOG.md`
- `docs/recipes/flash-next-cuda.md`
- `docs/vision.md`
- `zig/build/cuda_kernels.zig`
- `zig/kernels/cuda/contract_controls.cu`
- `zig/kernels/cuda/flashnext_attn.cu`
- `zig/kernels/cuda/flashnext_batch.cu`
- `zig/kernels/cuda/flashnext_hc.cu`
- `zig/kernels/cuda/flashnext_hcf.cu`
- `zig/kernels/cuda/flashnext_mtp.cu`
- `zig/kernels/cuda/nvfp4_experts_tile.cu`
- `zig/kernels/cuda/qmmf.cu`
- `zig/src/core/lanes/allocate.zig`
- `zig/src/core/lanes/config.zig`
- `zig/src/core/lanes/depth.zig`
- `zig/src/core/lanes/draft_row_test.zig`
- `zig/src/core/lanes/engine.zig`
- `zig/src/core/lanes/lanes.zig`
- `zig/src/core/lanes/windows.zig`
- `zig/src/cuda/kernels.zig`
- `zig/src/cuda/nvfp4_tile.zig`
- `zig/src/cuda/qmmf.zig`
- `zig/src/cuda/root.zig`
- `zig/src/families/flashnext/cuda_batch.zig`
- `zig/src/families/flashnext/cuda_batch_lanes.zig`
- `zig/src/families/flashnext/cuda_batch_moe.zig`
- `zig/src/families/flashnext/cuda_batch_side.zig`
- `zig/src/families/flashnext/cuda_batch_state.zig`
- `zig/src/families/flashnext/cuda_gdn.zig`
- `zig/src/families/flashnext/cuda_glue.zig`
- `zig/src/families/flashnext/cuda_host_draw.zig`
- `zig/src/families/flashnext/cuda_images.zig`
- `zig/src/families/flashnext/cuda_knobs.zig`
- `zig/src/families/flashnext/cuda_kv.zig`
- `zig/src/families/flashnext/cuda_lanes.zig`
- `zig/src/families/flashnext/cuda_lanes_draft.zig`
- `zig/src/families/flashnext/cuda_lanes_explain.zig`
- `zig/src/families/flashnext/cuda_lanes_load.zig`
- `zig/src/families/flashnext/cuda_lanes_verify.zig`
- `zig/src/families/flashnext/cuda_launch.zig`
- `zig/src/families/flashnext/cuda_media.zig`
- `zig/src/families/flashnext/cuda_mtp.zig`
- `zig/src/families/flashnext/cuda_mtp_draft.zig`
- `zig/src/families/flashnext/cuda_mtp_load.zig`
- `zig/src/families/flashnext/cuda_mtp_ops.zig`
- `zig/src/families/flashnext/cuda_ple.zig`
- `zig/src/families/flashnext/cuda_ple_nvfp4_test.zig`
- `zig/src/families/flashnext/cuda_ple_reader.zig`
- `zig/src/families/flashnext/cuda_ple_reader_test.zig`
- `zig/src/families/flashnext/cuda_prompt.zig`
- `zig/src/families/flashnext/cuda_prompt_chunks.zig`
- `zig/src/families/flashnext/cuda_round_costs.zig`
- `zig/src/families/flashnext/cuda_snap.zig`
- `zig/src/families/flashnext/cuda_tile_kernels.zig`
- `zig/src/families/flashnext/cuda_tiles.zig`
- `zig/src/native/cuda.zig`
- `zig/src/native/cuda_flashnext_kv8_test.zig`
- `zig/src/native/cuda_startup_test.zig`
- `zig/src/vision/qwen_cuda_dequant.zig`
- `zig/src/vision/qwen_cuda_media.zig`
- `zig/src/vision/qwen_cuda_weights.zig`
- `zig/src/vision/qwen_cuda_weights_logs.zig`
- `zig/src/vision/qwen_cuda_weights_quant_test.zig`
- `zig/src/vision/qwen_cuda_weights_test.zig`
- `zig/tests/cuda/flashnext_cand_tests.zig`
- `zig/tests/cuda/flashnext_hc_fuse_tests.zig`
- `zig/tests/cuda/flashnext_keep_late_tests.zig`
- `zig/tests/cuda/flashnext_kv8_tests.zig`
- `zig/tests/cuda/flashnext_mtp_parity_tests.zig`
- `zig/tests/cuda/main.zig`
- `zig/tests/cuda/nvfp4_experts_tile_tests.zig`
- `zig/tests/cuda/w4a8_tests.zig`

## This fork's README

A section on top of TensorFold's README says what this fork is, how the hill climb runs and where its results go, with the collaboration's image (new). The files:

- `README.md`
- `assets/mia-tensorfold-thermite.png` (new)

## The hill climb's changes

Each later commit is a change found by a hill climb run with [Thermite](https://github.com/iammrduncan/thermite) on DGX Sparks, confirmed by its judge, and states in its message what it changes and what it measured. Each is a patch on top of the recipe's, in its form: `git diff` from the commit before it.
