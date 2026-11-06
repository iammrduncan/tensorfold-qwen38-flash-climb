//! prompt_logprobs on Flash Next's CUDA lanes: every prompt row's scores through the head, a chunk at a time, read by
//! the shared logprob kernel (sample.cu's tf_logprob_rows, a fixed reduction order) into the stream's prompt rows: the
//! next prompt token's log probability and rank, and the `k` best tokens (vLLM's prompt_logprobs).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const prompt = @import("cuda_prompt.zig");

const LogRow = lanes.LogprobRow;
const row_words = lanes.logprob.words;

/// Rows a prompt_logprobs stream's chunk takes at most: their scores are [rows, vocab] bf16, about 254 MB.
pub const max_rows: usize = 512;

/// The kernel, a chunk's scores and their words on the host, made at the first stream that asks.
pub const Rows = struct {
    module: cuda.Module,
    draws: cuda.sample.Draws,
    logits: cuda.DeviceBuffer,
    /// Host-mapped: max_rows rows of row_words, then max_rows picks.
    out: cuda.HostBuffer,
    out_dev: u64,

    pub fn init(d: *const cuda.Driver) !Rows {
        var module = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer module.unload();
        var logits = try cuda.DeviceBuffer.alloc(d, max_rows * prompt.vocabulary * 2);
        errdefer logits.free();
        var out = try cuda.HostBuffer.allocMapped(d, max_rows * (row_words + 1) * 4);
        errdefer out.free();
        return .{ .module = module, .draws = try cuda.sample.Draws.resolve(module), .logits = logits, .out = out, .out_dev = try out.device() };
    }

    pub fn deinit(r: *Rows) void {
        r.out.free();
        r.logits.free();
        r.module.unload();
    }

    /// The rows of a chunk of `n` prompt rows from `at`, whose scores the pass left in `r.logits`: row i predicts
    /// `ids[at + i + 1]`. The prompt's last row predicts the reply's first token, which is not a prompt row.
    pub fn take(r: *Rows, stream: cuda.Stream, gpa: std.mem.Allocator, s: *lanes.Stream, ids: []const u32, at: usize, n: usize, k: u8) !void {
        if (n > max_rows) return error.UnexpectedTensor;
        const count = @min(n, ids.len - 1 - at);
        if (count == 0) return;
        const words = r.out.slice(u32);
        const picks_at = max_rows * row_words;
        @memcpy(words[picks_at..][0..count], ids[at + 1 ..][0..count]);
        try r.draws.logprobRows(stream, r.logits.ptr, prompt.vocabulary, r.out_dev + picks_at * 4, k, true, r.out_dev, count);
        try stream.synchronize();
        try s.prompt_rows.ensureUnusedCapacity(gpa, count);
        for (0..count) |i| s.prompt_rows.appendAssumeCapacity(LogRow.fromWords(words[i * row_words ..][0..row_words], ids[at + 1 + i], k));
    }
};
