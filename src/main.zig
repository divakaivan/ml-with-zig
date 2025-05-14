const std = @import("std");

const train = [_][2]f32{
    [_]f32{ 0.0, 0.0 },
    [_]f32{ 1.0, 2.0 },
    [_]f32{ 2.0, 4.0 },
    [_]f32{ 3.0, 6.0 },
    [_]f32{ 4.0, 8.0 },
};

const train_count = train.len / train[0].len;

fn randFloat() f32 {
    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();

    return rand.float(f32);
}

fn cost(w: f32) f32 {
    var result: f32 = 0.0;
    for (train) |elem| {
        const x = elem[0];
        const y = x * w;
        const d = y - elem[1];
        result += d * d;
    }
    result /= train_count;
    return result;
}

fn dcost(w: f32) f32 {
    var result: f32 = 0.0;
    for (train) |elem| {
        const x = elem[0];
        const y = elem[1];
        result += 2 * (x * w - y) * x;
    }
    result /= train_count;
    return result;
}

pub fn main() !void {
    var w = randFloat() * 10.0;
    const lr = 1e-2;
    std.debug.print("cost = {d}, w = {d}\n", .{ cost(w), w });
    for (0..20) |i| {
        var dw: f32 = 0.0;
        if (i == 0) {
            const eps = 1e-3;
            const c = cost(w);
            dw = (cost(w + eps) - c) / eps;
        } else {
            dw = dcost(w);
        }
        w -= lr * dw;
        std.debug.print("cost = {d}, w = {d}\n", .{ cost(w), w });
    }
    std.debug.print("---------------------------\n", .{});
    std.debug.print("w = {d}", .{w});
}
