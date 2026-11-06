//! The first token's logprob row on CUDA (OpenAI logprobs): the prompt's last row of scores, the one the first token
//! is drawn from, read on the host in f64. No kernel: the row is already in seq.logits.
const std = @import("std");
const lanes = @import("lanes");

const Row = lanes.LogprobRow;
const max_top = lanes.logprob.max_top;

fn at(logits: []const u8, i: usize) f64 {
    const bits: u32 = std.mem.readInt(u16, logits[2 * i ..][0..2], .little);
    return @as(f32, @bitCast(bits << 16));
}

/// `token`'s row in `logits` (bf16, little-endian, one a vocabulary id): its log probability, the best `count` ids by
/// score with theirs (ties by the lower id, as argmax breaks them), and its rank (the vocabulary's ids scored at or
/// above it, itself included). A non-finite score is refused.
pub fn of(logits: []const u8, token: u32, count: u8) !Row {
    const n = logits.len / 2;
    if (n == 0 or logits.len % 2 != 0 or token >= n or count > max_top) return error.UnexpectedTensor;
    var top: f64 = -std.math.inf(f64);
    for (0..n) |i| {
        const v = at(logits, i);
        if (!std.math.isFinite(v)) return error.NonFiniteScore;
        top = @max(top, v);
    }
    var sum: f64 = 0;
    for (0..n) |i| sum += @exp(at(logits, i) - top);
    const lse = top + @log(sum);
    const pick = at(logits, token);
    var ids: [max_top]u32 = undefined;
    var best: [max_top]f64 = undefined;
    var have: usize = 0;
    var rank: u32 = 0;
    for (0..n) |i| {
        const v = at(logits, i);
        if (v >= pick) rank += 1;
        if (count == 0 or (have == count and !(v > best[have - 1]))) continue;
        var j = if (have < count) have else have - 1;
        if (have < count) have += 1;
        while (j > 0 and v > best[j - 1]) : (j -= 1) {
            best[j] = best[j - 1];
            ids[j] = ids[j - 1];
        }
        best[j] = v;
        ids[j] = @intCast(i);
    }
    var r: Row = .{ .token = token, .logprob = @floatCast(pick - lse), .count = count, .rank = rank };
    for (0..count) |i| {
        r.ids[i] = ids[i];
        r.logprobs[i] = @floatCast(best[i] - lse);
    }
    return r;
}

fn bf16(v: f32) [2]u8 {
    var out: [2]u8 = undefined;
    std.mem.writeInt(u16, &out, @truncate(@as(u32, @bitCast(v)) >> 16), .little);
    return out;
}

test "a row's log probabilities against an f64 log-softmax, the best ids with ties by the lower id, and the rank" {
    const scores = [_]f32{ 1.0, 3.0, -2.0, 3.0, 0.5, 2.0 };
    var raw: [scores.len * 2]u8 = undefined;
    for (scores, 0..) |v, i| @memcpy(raw[2 * i ..][0..2], &bf16(v));
    var sum: f64 = 0;
    for (scores) |v| sum += @exp(@as(f64, v) - 3.0);
    const lse = 3.0 + @log(sum);
    const r = try of(&raw, 5, 3);
    try std.testing.expectEqual(@as(u32, 5), r.token);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 - lse), r.logprob, 1e-6);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5 }, r.ids[0..3]);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0 - lse), r.logprobs[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 - lse), r.logprobs[2], 1e-6);
    try std.testing.expectEqual(@as(u32, 3), r.rank);
    // the best of all: rank 2, with its tie at the same score
    try std.testing.expectEqual(@as(u32, 2), (try of(&raw, 3, 0)).rank);
    try std.testing.expectEqual(@as(u8, 0), (try of(&raw, 3, 0)).count);
}

test "a non-finite score, a token outside the row and too many best ids are refused" {
    var raw: [4]u8 = undefined;
    @memcpy(raw[0..2], &bf16(1.0));
    @memcpy(raw[2..4], &bf16(std.math.inf(f32)));
    try std.testing.expectError(error.NonFiniteScore, of(&raw, 0, 1));
    @memcpy(raw[2..4], &bf16(0.0));
    try std.testing.expectError(error.UnexpectedTensor, of(&raw, 2, 1));
    try std.testing.expectError(error.UnexpectedTensor, of(&raw, 0, max_top + 1));
}
