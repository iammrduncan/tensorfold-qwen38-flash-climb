# Modifications

This branch is [TensorFold](https://github.com/ashhart/TensorFold), release v1.2.3 (commit `6ec5fa4cb233f3606acbdad5bec433f0f8758f96`), by Ash Hart and the TensorFold contributors, under its licences (`LICENSE`, `NOTICE`, `LICENSES/`), with the changes below. As the Apache License 2.0 asks, this file states which files were changed.

It is one of two starting points of a [Thermite](https://github.com/iammrduncan/thermite) hill climb of Qwen3.8 Flash Next on one DGX Spark, with [Mia AI Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) as the other (this repository's default branch, `qwen38-flash-climb`: TensorFold v1.1.4 with the recipe's 17 patches). Here, TensorFold v1.2.3 runs as released, on its own engine settings; the recipe's patches, written for v1.1.4, are the climb's reference, ported where they still pay.

## This branch's own files

- `MODIFICATIONS.md` (new)

## The hill climb's changes

Each later commit is a change found by a hill climb run with Thermite on DGX Sparks, confirmed by its judge, and states in its message what it changes and what it measured.
