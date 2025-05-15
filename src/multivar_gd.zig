const std = @import("std");

const train = [_][3]f32{
    [_]f32{ 0.0, 0.0, 0.0 },
    [_]f32{ 1.0, 2.0, 8.0 },
    [_]f32{ 2.0, 4.0, 16.0 },
    [_]f32{ 3.0, 6.0, 24.0 },
    [_]f32{ 4.0, 8.0, 32.0 },
};

fn randFloat(seed: u64) f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    return rand.float(f32);
}

fn dot(x1: f32, x2: f32, w1: f32, w2: f32) f32 {
    return x1 * w1 + x2 * w2;
}

pub fn main() !void {
    var w1: f32 = randFloat(2);
    var w2: f32 = randFloat(4);

    for (0..2000) |epoch| {
        var total_loss: f32 = 0.0;
        var dw1: f32 = 0.0;
        var dw2: f32 = 0.0;

        // todo: maybe use @Vector
        for (train) |item| {
            const x1 = item[0];
            const x2 = item[1];
            const y_true = item[2];

            const y_pred = dot(x1, x2, w1, w2);
            const loss = (y_pred - y_true) * (y_pred - y_true);
            total_loss += loss;

            dw1 += 2 * x1 * (y_pred - y_true);
            dw2 += 2 * x2 * (y_pred - y_true);
        }
        w1 -= 0.003 * dw1 / train.len;
        w2 -= 0.003 * dw2 / train.len;

        if (epoch % 100 == 0) {
            std.debug.print("Epoch {d}, Loss {d}, W1 {d}, W2 {d}\n", .{ epoch, total_loss, w1, w2 });
        }
    }
    std.debug.print("\nResults\nW1 = {d}, W2 = {d}\n", .{ w1, w2 });
}
