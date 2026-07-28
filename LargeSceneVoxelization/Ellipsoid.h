#pragma once
#include "Types.h"
#include "Triangle.h"
#include "Polygon.h"

struct Ellipsoid {
    float3 center;
    float3x3 B;  // x^T B x <= 1

    // --- Jacobi eigen-decomposition for symmetric 3x3 ---
    static void eigenSym3_Jacobi(const float3x3& A_in, float3x3& R, float3& eval) {
        float3x3 A = 0.5f * (A_in + transpose(A_in));
        R = identity3x3();
        for (int sweep = 0; sweep < 8; ++sweep) {
            auto jacobiRotate = [&](int p, int q) {
                float apq = A[p][q];
                if (std::abs(apq) < 1e-10f) return;
                float tau = (A[q][q] - A[p][p]) / (2.0f * apq);
                float t = (tau >= 0.0f) ? 1.0f / (tau + std::sqrt(1.0f + tau * tau))
                                        : -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
                float c = 1.0f / std::sqrt(1.0f + t * t);
                float s = t * c;
                for (int k = 0; k < 3; ++k) {
                    if (k == p || k == q) continue;
                    float akp = A[k][p], akq = A[k][q];
                    A[k][p] = A[p][k] = c * akp - s * akq;
                    A[k][q] = A[q][k] = s * akp + c * akq;
                }
                float app = A[p][p], aqq = A[q][q];
                A[p][p] = c*c*app - 2.0f*s*c*apq + s*s*aqq;
                A[q][q] = s*s*app + 2.0f*s*c*apq + c*c*aqq;
                A[p][q] = A[q][p] = 0.0f;
                for (int i = 0; i < 3; ++i) {
                    float vip = R[p][i], viq = R[q][i];
                    R[p][i] = c * vip - s * viq;
                    R[q][i] = s * vip + c * viq;
                }
            };
            jacobiRotate(0, 1); jacobiRotate(0, 2); jacobiRotate(1, 2);
        }
        eval = float3(A[0][0], A[1][1], A[2][2]);
        // Sort descending
        auto swapCol = [&](int a, int b) {
            for (int r = 0; r < 3; ++r) std::swap(R[a][r], R[b][r]);
            std::swap(eval[a], eval[b]);
        };
        if (eval.x < eval.y) swapCol(0, 1);
        if (eval.x < eval.z) swapCol(0, 2);
        if (eval.y < eval.z) swapCol(1, 2);
    }

    // --- Fit ellipsoid to polygons (ported from Ellipsoid.slang) ---
    void fit(const std::vector<Polygon>& polygons, const PolygonRange& range) {
        float totalArea = 0.0f;
        center = range.calcCentroid(polygons, totalArea);
        if (totalArea <= 0.0f) { B = zeros3x3(); center = float3(0); return; }

        // Area-weighted covariance
        float3x3 cov = zeros3x3();
        for (uint pi = 0; pi < range.count; ++pi) {
            const Polygon& poly = polygons[pi + range.localHead];
            const float3& v0 = poly.vertices[0];
            for (uint i = 1; i + 1 < poly.count; ++i) {
                float3 a = v0, b = poly.vertices[i], c = poly.vertices[i + 1];
                float3 ab = b - a, ac = c - a;
                float3 s = cross(ab, ac);
                float area = safeSqrt(dot(s, s)) * 0.5f;
                if (area <= 0) continue;
                float3 a0 = a - center, b0 = b - center, c0 = c - center;
                float3 S1 = a0 + b0 + c0;
                float w = area / 12.0f;
                auto addCov = [&](int i, int j, float v) { cov[i][j] += w * v; };
                addCov(0,0, S1.x*S1.x + a0.x*a0.x + b0.x*b0.x + c0.x*c0.x);
                addCov(0,1, S1.x*S1.y + a0.x*a0.y + b0.x*b0.y + c0.x*c0.y);
                addCov(0,2, S1.x*S1.z + a0.x*a0.z + b0.x*b0.z + c0.x*c0.z);
                addCov(1,0, S1.y*S1.x + a0.y*a0.x + b0.y*b0.x + c0.y*c0.x);
                addCov(1,1, S1.y*S1.y + a0.y*a0.y + b0.y*b0.y + c0.y*c0.y);
                addCov(1,2, S1.y*S1.z + a0.y*a0.z + b0.y*b0.z + c0.y*c0.z);
                addCov(2,0, S1.z*S1.x + a0.z*a0.x + b0.z*b0.x + c0.z*c0.x);
                addCov(2,1, S1.z*S1.y + a0.z*a0.y + b0.z*b0.y + c0.z*c0.y);
                addCov(2,2, S1.z*S1.z + a0.z*a0.z + b0.z*b0.z + c0.z*c0.z);
            }
        }
        cov = mul(1.0f / totalArea, cov);
        cov = 0.5f * (cov + transpose(cov));
        float tr = cov[0][0] + cov[1][1] + cov[2][2];
        float lam = 1e-6f * std::max(tr, 1e-6f);
        cov[0][0] += lam; cov[1][1] += lam; cov[2][2] += lam;

        float3x3 R; float3 eval;
        eigenSym3_Jacobi(cov, R, eval);

        float3 center0 = center;
        // PCA-space extents
        float3 qMin(1e30f), qMax(-1e30f);
        float3 qMinPt[3] = {}, qMaxPt[3] = {};
        for (uint pi = 0; pi < range.count; ++pi) {
            const Polygon& poly = polygons[pi + range.localHead];
            for (uint i = 0; i < poly.count; ++i) {
                float3 q = transpose(R) * (poly.vertices[i] - center0);
                if (q.x < qMin.x) { qMin.x = q.x; qMinPt[0] = q; }
                if (q.x > qMax.x) { qMax.x = q.x; qMaxPt[0] = q; }
                if (q.y < qMin.y) { qMin.y = q.y; qMinPt[1] = q; }
                if (q.y > qMax.y) { qMax.y = q.y; qMaxPt[1] = q; }
                if (q.z < qMin.z) { qMin.z = q.z; qMinPt[2] = q; }
                if (q.z > qMax.z) { qMax.z = q.z; qMaxPt[2] = q; }
            }
        }

        float3 shift = 0.5f * (qMin + qMax);
        float3 half  = glm::max(0.5f * (qMax - qMin), float3(1e-3f));
        float3 invHalf = 1.0f / half;

        // Ritter bounding sphere in normalized PCA space
        float3 yExt[6] = {
            (qMinPt[0] - shift) * invHalf, (qMaxPt[0] - shift) * invHalf,
            (qMinPt[1] - shift) * invHalf, (qMaxPt[1] - shift) * invHalf,
            (qMinPt[2] - shift) * invHalf, (qMaxPt[2] - shift) * invHalf,
        };
        int ia = 0, ib = 1;
        float bestD2 = -1;
        for (int a = 0; a < 6; ++a)
            for (int b = a + 1; b < 6; ++b) {
                float d2 = dot(yExt[b] - yExt[a], yExt[b] - yExt[a]);
                if (d2 > bestD2) { bestD2 = d2; ia = a; ib = b; }
            }

        float3 sphC = 0.5f * (yExt[ia] + yExt[ib]);
        float  sphR = glm::length(yExt[ib] - sphC);
        sphR = std::max(sphR, 1e-8f);

        for (uint pi = 0; pi < range.count; ++pi) {
            const Polygon& poly = polygons[pi + range.localHead];
            for (uint i = 0; i < poly.count; ++i) {
                float3 y = (transpose(R) * (poly.vertices[i] - center0) - shift) * invHalf;
                float3 d = y - sphC;
                float dist2 = dot(d, d);
                if (dist2 > sphR * sphR) {
                    float dist = std::sqrt(dist2);
                    float newR = 0.5f * (sphR + dist);
                    sphC += d * ((newR - sphR) / std::max(dist, 1e-12f));
                    sphR = newR;
                }
            }
        }
        // Tighten
        float r2 = 0;
        for (uint pi = 0; pi < range.count; ++pi) {
            const Polygon& poly = polygons[pi + range.localHead];
            for (uint i = 0; i < poly.count; ++i) {
                float3 y = (transpose(R) * (poly.vertices[i] - center0) - shift) * invHalf;
                r2 = std::max(r2, dot(y - sphC, y - sphC));
            }
        }
        sphR = std::sqrt(r2) * (1.0f + 1e-6f);

        float3 qCenter = shift + half * sphC;
        center = center0 + R * qCenter;
        float3 axis = half * sphR;
        axis = glm::max(axis, float3(1e-3f));
        float3 invAxis2 = 1.0f / (axis * axis);
        B = R * diag3(invAxis2) * transpose(R);
        B = 0.5f * (B + transpose(B));
        center = center - float3(range.cellInt);
    }

    // Fit an ellipsoid from a set of 3D points (non-area-weighted).
    // Used for parent node ellipsoids aggregated from child ellipsoid extreme points.
    // Points must be in normalized [0, 1] space relative to cellInt.
    void fitFromPoints(const std::vector<float3>& points, const int3& cellInt) {
        if (points.size() < 4) {
            center = float3(0); B = zeros3x3();
            return;
        }

        // Centroid (simple average)
        float3 centroid(0);
        for (auto& p : points) centroid += p;
        centroid /= (float)points.size();

        // Covariance matrix
        float3x3 cov = zeros3x3();
        for (auto& p : points) {
            float3 d = p - centroid;
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    cov[i][j] += d[i] * d[j];
        }
        cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];
        cov = mul(1.0f / (float)points.size(), cov);
        cov = 0.5f * (cov + transpose(cov));

        float tr = cov[0][0] + cov[1][1] + cov[2][2];
        float lam = 1e-6f * std::max(tr, 1e-6f);
        cov[0][0] += lam; cov[1][1] += lam; cov[2][2] += lam;

        float3x3 R;
        float3 evals;
        eigenSym3_Jacobi(cov, R, evals);

        // PCA-space extents
        float3 qMin(1e30f), qMax(-1e30f);
        float3 qMinPt[3] = {}, qMaxPt[3] = {};
        for (auto& p : points) {
            float3 q = transpose(R) * (p - centroid);
            if (q.x < qMin.x) { qMin.x = q.x; qMinPt[0] = q; }
            if (q.x > qMax.x) { qMax.x = q.x; qMaxPt[0] = q; }
            if (q.y < qMin.y) { qMin.y = q.y; qMinPt[1] = q; }
            if (q.y > qMax.y) { qMax.y = q.y; qMaxPt[1] = q; }
            if (q.z < qMin.z) { qMin.z = q.z; qMinPt[2] = q; }
            if (q.z > qMax.z) { qMax.z = q.z; qMaxPt[2] = q; }
        }

        float3 shift = 0.5f * (qMin + qMax);
        float3 half = glm::max(0.5f * (qMax - qMin), float3(1e-3f));
        float3 invHalf = 1.0f / half;

        // Ritter bounding sphere in normalized PCA space
        float3 yExt[6] = {
            (qMinPt[0] - shift) * invHalf, (qMaxPt[0] - shift) * invHalf,
            (qMinPt[1] - shift) * invHalf, (qMaxPt[1] - shift) * invHalf,
            (qMinPt[2] - shift) * invHalf, (qMaxPt[2] - shift) * invHalf,
        };
        int ia = 0, ib = 1;
        float bestD2 = 0;
        for (int a = 0; a < 6; ++a)
            for (int b = a + 1; b < 6; ++b) {
                float d2 = dot(yExt[b] - yExt[a], yExt[b] - yExt[a]);
                if (d2 > bestD2) { bestD2 = d2; ia = a; ib = b; }
            }

        float3 sphC = 0.5f * (yExt[ia] + yExt[ib]);
        float sphR = std::max(glm::length(yExt[ib] - sphC), 1e-8f);

        for (auto& p : points) {
            float3 y = (transpose(R) * (p - centroid) - shift) * invHalf;
            float3 d = y - sphC;
            float dist2 = dot(d, d);
            if (dist2 > sphR * sphR) {
                float dist = std::sqrt(dist2);
                float newR = 0.5f * (sphR + dist);
                sphC += d * ((newR - sphR) / std::max(dist, 1e-12f));
                sphR = newR;
            }
        }

        // Tighten
        float r2 = 0;
        for (auto& p : points) {
            float3 y = (transpose(R) * (p - centroid) - shift) * invHalf;
            r2 = std::max(r2, dot(y - sphC, y - sphC));
        }
        sphR = std::sqrt(r2) * (1.0f + 1e-6f);

        // Construct ellipsoid
        float3 qCenter = shift + half * sphC;
        center = centroid + R * qCenter;
        float3 axis = half * sphR;
        axis = glm::max(axis, float3(1e-3f));
        float3 invAxis2 = 1.0f / (axis * axis);
        B = R * diag3(invAxis2) * transpose(R);
        B = 0.5f * (B + transpose(B));
        // Points are already in [0,1] relative to cellInt, no subtraction needed
        (void)cellInt;  // kept for API compatibility
    }

    // Clip a ray segment against the ellipsoid, returns (tEnter, tExit) in [0,1]
    float2 clip(const float3& from, const float3& to) const {
        float3 v = to - from;
        float3 f = from - center;
        float a = dot(v, B * v);
        float b = 2.0f * dot(v, B * f);
        float c = dot(f, B * f) - 1.0f;
        float d = b * b - 4.0f * a * c;
        if (d < 0) return float2(-1, -1);
        float s = std::sqrt(d);
        float uIn = std::max(0.0f, (-b - s) / (2.0f * a));
        float uOut = std::min(1.0f, (-b + s) / (2.0f * a));
        if (uIn > uOut) return float2(-1, -1);
        return float2(uIn, uOut);
    }
};
