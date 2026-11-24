/*
 * Knob geometry adapted from sparse-conv-gpis-tungsten's SdfFunctions.
 * Copyright 2020 Towaki Takikawa. Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

RWTexture3D<float> gValues;
uniform float3 gOrigin;
uniform float gSpacing;
uniform uint gSize;

float smoothUnion(float a, float b, float width)
{
    float h = saturate(0.5f + 0.5f * (b - a) / width);
    return lerp(b, a, h) - width * h * (1.f - h);
}

float cylinder(float3 p, float radius, float halfHeight)
{
    float2 q = float2(length(p.xz) - radius, abs(p.y) - halfHeight);
    return min(max(q.x, q.y), 0.f) + length(max(q, 0.f));
}

float cone(float3 p)
{
    return (length(p.xy) + p.z) * (3.14159265359f / 3.f);
}

float prism(float3 p)
{
    return max(abs(p.z) - 0.3f, max(abs(p.x) * 0.866025f + p.y * 0.5f, -p.y) - 0.6f);
}

float shaderball(float3 world)
{
    float3 p = (world - float3(0.f, 0.8f, 0.f)) / 0.8f;
    float outer = length(p) - 1.f;
    float cut = length(p - float3(0.f, 0.5f, 0.5f)) - 0.7f;
    float3 t = p - float3(0.f, 0.2f, 0.2f);
    float3 torus = float3(t.x, 0.70710678f * (t.y + t.z), 0.70710678f * (t.z - t.y));
    float etch = length(float2(length(torus.xz) - 1.f, torus.y)) - 0.05f;

    float shell = -smoothUnion(cut, -outer, 0.1f);
    float ball = max(-etch, min(shell, length(p) - 0.75f));

    float3 b = p + float3(0.f, 0.775f, 0.f);
    float base = smoothUnion(cone(float3(b.x, b.z, -b.y - 0.9f)), cone(float3(b.x, -b.z, b.y - 0.9f)), 0.02f);
    base = max(base, cylinder(b, 1.1f, 0.25f)) * 0.7f;
    base = max(-cylinder(b, 0.6f, 0.3f), base);
    base = max(-prism(float3(b.x, 1.f - b.z, b.y)), base);
    return 0.8f * min(-smoothUnion(outer, -base, 0.1f), ball);
}

[numthreads(8, 8, 4)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (any(id >= gSize)) return;
    gValues[id] = shaderball(gOrigin + float3(id) * gSpacing);
}
