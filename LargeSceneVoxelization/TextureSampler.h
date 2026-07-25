#pragma once
#include "Types.h"
#include <vector>
#include <string>
#include <cmath>

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"

// Simple CPU MIP-mapped texture for LOD sampling (stores linear float values)
struct Texture2D {
    std::vector<std::vector<float>> mipLevels;  // each level: RGBA float row-major
    int width = 0, height = 0;
    int levels = 0;

    bool load(const std::string& path, bool srgb = false) {
        int w, h, comp;
        uint8_t* data = stbi_load(path.c_str(), &w, &h, &comp, 4);
        if (!data) return false;
        width = w; height = h;
        levels = (int)std::floor(std::log2(std::max(w, h))) + 1;
        mipLevels.resize(levels);

        // Level 0: convert uint8 → linear float
        mipLevels[0].resize(w * h * 4);
        if (srgb) {
            for (int i = 0; i < w * h * 4; i += 4) {
                mipLevels[0][i]   = srgbChannelToLinear(data[i]   / 255.0f);
                mipLevels[0][i+1] = srgbChannelToLinear(data[i+1] / 255.0f);
                mipLevels[0][i+2] = srgbChannelToLinear(data[i+2] / 255.0f);
                mipLevels[0][i+3] = data[i+3] / 255.0f;
            }
        } else {
            for (int i = 0; i < w * h * 4; i++) {
                mipLevels[0][i] = data[i] / 255.0f;
            }
        }
        stbi_image_free(data);

        // Build MIP chain with box filtering (linear float space, no quantization loss)
        for (int l = 1; l < levels; l++) {
            int pw = std::max(1, width >> (l - 1));
            int ph = std::max(1, height >> (l - 1));
            int cw = std::max(1, width >> l);
            int ch = std::max(1, height >> l);
            auto& prev = mipLevels[l - 1];
            mipLevels[l].resize(cw * ch * 4);
            for (int y = 0; y < ch; y++) {
                for (int x = 0; x < cw; x++) {
                    float sum[4] = {};
                    int sx0 = x * 2, sy0 = y * 2;
                    for (int dy = 0; dy < 2; dy++) {
                        for (int dx = 0; dx < 2; dx++) {
                            int sx = std::min(sx0 + dx, pw - 1);
                            int sy = std::min(sy0 + dy, ph - 1);
                            int idx = (sy * pw + sx) * 4;
                            for (int c = 0; c < 4; c++)
                                sum[c] += prev[idx + c];
                        }
                    }
                    int idx = (y * cw + x) * 4;
                    for (int c = 0; c < 4; c++)
                        mipLevels[l][idx + c] = sum[c] / 4.0f;
                }
            }
        }
        return true;
    }

    // Sample at given MIP level with bilinear filtering (returns linear float)
    float4 sample(int level, float2 uv) const {
        level = std::max(0, std::min(level, levels - 1));
        int w = std::max(1, width >> level);
        int h = std::max(1, height >> level);
        const float* data = mipLevels[level].data();

        uv = uv - glm::floor(uv);  // fract → [0, 1)
        float fx = uv.x * (float)w - 0.5f;
        float fy = uv.y * (float)h - 0.5f;
        int x0 = (int)std::floor(fx);
        int y0 = (int)std::floor(fy);
        float tx = fx - (float)x0;
        float ty = fy - (float)y0;
        x0 = ((x0 % w) + w) % w;
        y0 = ((y0 % h) + h) % h;
        int x1 = (x0 + 1) % w;
        int y1 = (y0 + 1) % h;

        auto fetch = [&](int x, int y) -> float4 {
            int idx = (y * w + x) * 4;
            return float4(data[idx], data[idx+1], data[idx+2], data[idx+3]);
        };

        float4 v00 = fetch(x0, y0), v10 = fetch(x1, y0);
        float4 v01 = fetch(x0, y1), v11 = fetch(x1, y1);
        return lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty);
    }
};

// Area-weighted texture LOD sampling (matches sampleTextureArea in AnalyzePolygon.cs.slang)
inline float4 sampleTextureArea(const Texture2D& tex, const float2& uv, float uvArea,
                                 const float4& uniformValue) {
    if (tex.width == 0) return uniformValue;
    float pixelArea = uvArea * tex.width * tex.height;
    float lodLevel = 0.5f * std::log2(std::max(pixelArea, 1.0f));
    // Trilinear: blend between floor and ceil MIP levels
    int lodLo = std::max(0, std::min((int)std::floor(lodLevel), tex.levels - 1));
    int lodHi = std::max(0, std::min(lodLo + 1, tex.levels - 1));
    float lodFrac = lodLevel - (float)lodLo;
    float4 vLo = tex.sample(lodLo, uv);
    float4 vHi = tex.sample(lodHi, uv);
    return lerp(vLo, vHi, lodFrac);
}
