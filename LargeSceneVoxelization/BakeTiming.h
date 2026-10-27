#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace BakeTiming
{
using Clock = std::chrono::steady_clock;

inline double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Coarse wall-clock measurements are always enabled. next() closes the previous
// stage, so stages in a pipeline do not overlap, including independent LODs.
class Stage
{
public:
    explicit Stage(std::string label) : mLabel(std::move(label)), mStart(Clock::now()) {}
    ~Stage() { stop(); }
    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

    void stop()
    {
        if (mLabel.empty()) return;
        const double elapsed = secondsSince(mStart);
        std::ostringstream out;
        out << "  [Timing] " << mLabel << " wall=" << std::fixed << std::setprecision(6)
            << elapsed << " s\n";
        std::cout << out.str() << std::flush;
        mLabel.clear();
    }

    void next(std::string label)
    {
        stop();
        mLabel = std::move(label);
        mStart = Clock::now();
    }

private:
    std::string mLabel;
    Clock::time_point mStart;
};

enum class Task : size_t
{
    PolygonRead,
    NodeCopy,
    PrepareGeometry,
    MaterialInputs,
    EllipsoidFit,
    PrimitiveProjection,
    VisibleProjection,
    TotalProjection,
    VisibleSetup,
    VisibleRaster,
    VisibleAccumulate,
    VisibleFinalize,
    MaskSetup,
    MaskRaster,
    MaskSampling,
    ParentAggregation,
    Count
};

inline constexpr std::array<const char*, size_t(Task::Count)> TaskNames = {
    "polygon read/deserialize/allocate", "node polygon copy", "prepare geometry",
    "texture/material inputs + legacy ABSDF", "ellipsoid fit",
    "primitive projection samples", "legacy visible-area raster", "total projected area",
    "visible material setup", "visible material depth raster", "visible material pixel accumulation",
    "visible material finalize/pack", "mask setup", "mask depth raster + eligibility",
    "mask CDF/rejection/visibility", "approximate parent aggregation"
};

// Owned by exactly one worker, then reduced after join. No profiling atomics or
// shared counters in the hot loops. These are elapsed times, NOT OS CPU times.
struct alignas(64) Worker
{
    double wall = 0;
    std::array<double, size_t(Task::Count)> tasks{};
    uint64_t nodes = 0;
    uint64_t polygons = 0;
    uint64_t solidNodes = 0;
    uint64_t primitiveSamples = 0;
    uint64_t maskCandidates = 0;
    uint64_t maskRetainedSamples = 0;
    uint64_t maskActiveLobes = 0;
    uint64_t maskInvalidLobes = 0;
};

// A sequential timer may change categories, but never counts the same interval
// twice. A null worker performs no clock reads (normal non-profiled bakes).
class Scope
{
public:
    Scope(Worker* worker, Task task) : Scope(worker ? &worker->tasks[size_t(task)] : nullptr) {}
    explicit Scope(double* destination) : mDestination(destination)
    {
        if (mDestination) mStart = Clock::now();
    }
    ~Scope() { stop(); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    void stop()
    {
        if (!mDestination) return;
        *mDestination += secondsSince(mStart);
        mDestination = nullptr;
    }

    void next(Worker* worker, Task task)
    {
        if (!worker) return;
        const auto now = Clock::now();
        if (mDestination) *mDestination += std::chrono::duration<double>(now - mStart).count();
        mDestination = &worker->tasks[size_t(task)];
        mStart = now;
    }

private:
    double* mDestination;
    Clock::time_point mStart{};
};

inline void reportAnalyze(const std::string& label, const std::vector<Worker>& workers, double wall)
{
    if (workers.empty()) return;
    Worker total;
    double longest = 0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "  [Timing] Analyze " << label << " workers wall=" << wall << " s\n";
    for (size_t i = 0; i < workers.size(); ++i)
    {
        const auto& w = workers[i];
        total.wall += w.wall;
        longest = std::max(longest, w.wall);
        for (size_t t = 0; t < total.tasks.size(); ++t) total.tasks[t] += w.tasks[t];
        total.nodes += w.nodes;
        total.polygons += w.polygons;
        total.solidNodes += w.solidNodes;
        total.primitiveSamples += w.primitiveSamples;
        total.maskCandidates += w.maskCandidates;
        total.maskRetainedSamples += w.maskRetainedSamples;
        total.maskActiveLobes += w.maskActiveLobes;
        total.maskInvalidLobes += w.maskInvalidLobes;
        out << "    worker " << i << ": wall=" << w.wall << " s, nodes=" << w.nodes
            << ", polygons=" << w.polygons << ", solid=" << w.solidNodes << '\n';
    }
    out << "    summed worker elapsed=" << total.wall << " s, max/mean="
        << (total.wall > 0 ? longest * workers.size() / total.wall : 0) << '\n';
    out << "    task totals: summed worker seconds, % of summed worker elapsed (not CPU utilization)\n";
    double measured = 0;
    for (size_t t = 0; t < total.tasks.size(); ++t)
    {
        const double seconds = total.tasks[t];
        measured += seconds;
        out << "      " << TaskNames[t] << ": " << seconds << " s ("
            << (total.wall > 0 ? 100 * seconds / total.wall : 0) << "%)\n";
    }
    const double other = std::max(0.0, total.wall - measured);
    out << "      other / timer overhead: " << other << " s ("
        << (total.wall > 0 ? 100 * other / total.wall : 0) << "%)\n";
    out << "    analyzed nodes=" << total.nodes << ", polygons=" << total.polygons
        << ", solid=" << total.solidNodes << ", primitive samples=" << total.primitiveSamples << '\n';
    out << "    mask candidates=" << total.maskCandidates << ", retained samples=" << total.maskRetainedSamples
        << ", retained/candidates="
        << (total.maskCandidates ? 100.0 * total.maskRetainedSamples / total.maskCandidates : 0)
        << "%, invalid lobes=" << total.maskInvalidLobes << '/' << total.maskActiveLobes << '\n';
    std::cout << out.str() << std::flush;
}
} // namespace BakeTiming
