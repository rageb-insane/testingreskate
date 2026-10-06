#pragma once
// BC7 encoding (mode 6 only: one subset, RGBA 7.7.7.7 endpoints with a p-bit
// each, 4-bit indices). Mode 6 alone is a good fit for flat deck graphics and
// keeps the encoder small. Decoding goes through bcdec.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace bc7 {

inline constexpr int weights[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

struct Block {
    std::array<std::array<int, 4>, 2> endpoint{};  // 8-bit expanded values
    std::array<int, 2> pbit{};
    std::array<std::array<int, 4>, 2> quant{};     // 7-bit stored values
    std::array<int, 16> index{};
    double error{std::numeric_limits<double>::max()};
};

inline int interpolate(int e0, int e1, int w) { return ((64 - w) * e0 + w * e1 + 32) >> 6; }

// Quantises a float endpoint to 7 bits with the given p-bit.
inline void quantise(const double* value, int p, int* quant, int* expanded) {
    for (int c = 0; c < 4; ++c) {
        int q = static_cast<int>(std::lround((std::clamp(value[c], 0.0, 255.0) - p) / 2.0));
        q = std::clamp(q, 0, 127);
        quant[c] = q;
        expanded[c] = (q << 1) | p;
    }
}

inline double assign(const std::uint8_t (*px)[4], Block& block) {
    int palette[16][4];
    for (int i = 0; i < 16; ++i)
        for (int c = 0; c < 4; ++c)
            palette[i][c] = interpolate(block.endpoint[0][c], block.endpoint[1][c], weights[i]);
    double total = 0;
    for (int p = 0; p < 16; ++p) {
        int best = 0;
        long bestError = std::numeric_limits<long>::max();
        for (int i = 0; i < 16; ++i) {
            long e = 0;
            for (int c = 0; c < 4; ++c) {
                const long d = static_cast<long>(px[p][c]) - palette[i][c];
                e += d * d;
            }
            if (e < bestError) { bestError = e; best = i; }
        }
        block.index[static_cast<std::size_t>(p)] = best;
        total += static_cast<double>(bestError);
    }
    return total;
}

// Least-squares endpoints for fixed indices.
inline bool refit(const std::uint8_t (*px)[4], const Block& block, double* e0, double* e1) {
    double aa = 0, ab = 0, bb = 0, ax[4]{}, bx[4]{};
    for (int p = 0; p < 16; ++p) {
        const double w = weights[block.index[static_cast<std::size_t>(p)]] / 64.0;
        const double a = 1.0 - w, b = w;
        aa += a * a; ab += a * b; bb += b * b;
        for (int c = 0; c < 4; ++c) { ax[c] += a * px[p][c]; bx[c] += b * px[p][c]; }
    }
    const double det = aa * bb - ab * ab;
    if (std::abs(det) < 1e-9) return false;
    for (int c = 0; c < 4; ++c) {
        e0[c] = (bb * ax[c] - ab * bx[c]) / det;
        e1[c] = (aa * bx[c] - ab * ax[c]) / det;
    }
    return true;
}

inline Block encode_block(const std::uint8_t (*px)[4]) {
    // Principal axis of the block's colours.
    double mean[4]{};
    for (int p = 0; p < 16; ++p) for (int c = 0; c < 4; ++c) mean[c] += px[p][c] / 16.0;
    double cov[4][4]{};
    for (int p = 0; p < 16; ++p)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) cov[i][j] += (px[p][i] - mean[i]) * (px[p][j] - mean[j]);
    double axis[4] = {1, 1, 1, 1};
    for (int iteration = 0; iteration < 8; ++iteration) {
        double next[4]{};
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) next[i] += cov[i][j] * axis[j];
        double length = std::sqrt(next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3]);
        if (length < 1e-9) break;
        for (int i = 0; i < 4; ++i) axis[i] = next[i] / length;
    }
    double lo = std::numeric_limits<double>::max(), hi = std::numeric_limits<double>::lowest();
    for (int p = 0; p < 16; ++p) {
        double t = 0;
        for (int c = 0; c < 4; ++c) t += (px[p][c] - mean[c]) * axis[c];
        lo = std::min(lo, t); hi = std::max(hi, t);
    }
    double start0[4], start1[4];
    for (int c = 0; c < 4; ++c) { start0[c] = mean[c] + lo * axis[c]; start1[c] = mean[c] + hi * axis[c]; }

    Block best;
    for (int p0 = 0; p0 < 2; ++p0) {
        for (int p1 = 0; p1 < 2; ++p1) {
            double e0[4], e1[4];
            std::memcpy(e0, start0, sizeof e0);
            std::memcpy(e1, start1, sizeof e1);
            for (int pass = 0; pass < 3; ++pass) {
                Block candidate;
                candidate.pbit = {p0, p1};
                quantise(e0, p0, candidate.quant[0].data(), candidate.endpoint[0].data());
                quantise(e1, p1, candidate.quant[1].data(), candidate.endpoint[1].data());
                candidate.error = assign(px, candidate);
                if (candidate.error < best.error) best = candidate;
                if (candidate.error == 0 || !refit(px, candidate, e0, e1)) break;
            }
        }
    }
    // The anchor (pixel 0) index must have its top bit clear.
    if (best.index[0] >= 8) {
        std::swap(best.endpoint[0], best.endpoint[1]);
        std::swap(best.quant[0], best.quant[1]);
        std::swap(best.pbit[0], best.pbit[1]);
        for (auto& i : best.index) i = 15 - i;
    }
    return best;
}

class BitWriter {
public:
    void put(std::uint64_t value, int bits) {
        for (int i = 0; i < bits; ++i, ++position_)
            if ((value >> i) & 1) data_[static_cast<std::size_t>(position_ >> 3)] |= static_cast<std::uint8_t>(1u << (position_ & 7));
    }
    const std::array<std::uint8_t, 16>& data() const { return data_; }
private:
    std::array<std::uint8_t, 16> data_{};
    int position_{};
};

inline std::array<std::uint8_t, 16> pack(const Block& block) {
    BitWriter out;
    out.put(1u << 6, 7);  // mode 6
    for (int c = 0; c < 4; ++c) {
        out.put(static_cast<std::uint64_t>(block.quant[0][static_cast<std::size_t>(c)]), 7);
        out.put(static_cast<std::uint64_t>(block.quant[1][static_cast<std::size_t>(c)]), 7);
    }
    out.put(static_cast<std::uint64_t>(block.pbit[0]), 1);
    out.put(static_cast<std::uint64_t>(block.pbit[1]), 1);
    for (int p = 0; p < 16; ++p) out.put(static_cast<std::uint64_t>(block.index[static_cast<std::size_t>(p)]), p == 0 ? 3 : 4);
    return out.data();
}

// RGBA8 image (rows top to bottom) to BC7 blocks; edges are clamped.
inline std::vector<std::uint8_t> encode(const std::uint8_t* rgba, int width, int height) {
    const int bw = std::max(1, (width + 3) / 4), bh = std::max(1, (height + 3) / 4);
    std::vector<std::uint8_t> out(static_cast<std::size_t>(bw) * bh * 16);
    for (int by = 0; by < bh; ++by) {
        for (int bx = 0; bx < bw; ++bx) {
            std::uint8_t px[16][4];
            for (int y = 0; y < 4; ++y)
                for (int x = 0; x < 4; ++x) {
                    const int sx = std::min(bx * 4 + x, width - 1), sy = std::min(by * 4 + y, height - 1);
                    std::memcpy(px[y * 4 + x], rgba + (static_cast<std::size_t>(sy) * width + sx) * 4, 4);
                }
            const auto bytes = pack(encode_block(px));
            std::memcpy(out.data() + (static_cast<std::size_t>(by) * bw + bx) * 16, bytes.data(), 16);
        }
    }
    return out;
}

} // namespace bc7
