# The recipe's patches for v1.1.4, still to carry over

These are 14 of the 17 patches of MiaAI-Lab's recipe [Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (`patches/` at `20e15e41f2f514a0d287710f4665ee1b5009ba41`), by MiaAI-Lab under the MIT License (`LICENSES/MiaAI-Lab-MIT.txt`, `NOTICE-recipe`), copied as they are. They were written for TensorFold v1.1.4 (`eab85828`), and none of them applies to v1.2.3 as it is. The other three (0001 to 0003) are already in this branch's history: see `MODIFICATIONS.md`.

The recipe's own summary of each, from its README:

| Patch | What it does |
| --- | --- |
| `0004`, `0006` | prompt expert tile and prefill work: no drain between prompt chunks, vectorised hyper-connection kernels, a 128x128 MXFP8 prompt tile, 3,072-row chunks |
| `0005` | the fp8 attention reader keeps the next tile in registers; the MTP head's cache stays bf16 |
| `0007` to `0011` | sampled requests join the batched draft pass, GPU top-k candidates, a round-cost model (off), n-gram rows of all lanes gathered at once, host draws on worker threads |
| `0012`, `0013` | a partly rejected draft window no longer replays the DeltaNet state; the n-gram table is read with `pread` and read ahead instead of through an `mmap` |
| `0014`, `0015` | fused hyper-connection kernels, expert row tiles and MTP head pieces for prompt passes; a two-stage MX128 prompt tile |
| `0016` | the shared expert runs on a second stream in decode rounds; a shared round takes every lane's DeltaNet rows in three launches a layer |
| `0017` | the draft chain stops on the running product of its draft probabilities (the rule of the earlier build) |

v1.2.3 already holds parts of some of them in its own form: for example `TF_FLASHNEXT_SAMPLED_HEADS` (0007), `TF_FLASHNEXT_KEEP_LATE` and `fn_gdn_rows_late` (0012), `TF_FLASHNEXT_WARM_AHEAD` (0013), `TF_FLASHNEXT_SHARED_SIDE` and `TF_FLASHNEXT_GDN_LANES` (0016), and a negative `TF_FLASHNEXT_CONFIDENCE` as a running product (0017). Carry over what is still missing, in v1.2.3's code, and keep each patch's switch names: the engine is served with the recipe's settings (its `scripts/config.sh`), so a switch the patch reads is set as the recipe sets it.
