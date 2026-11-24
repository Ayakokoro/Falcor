#include "LatticeGPISPass.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderPassStandardFlags.h"

namespace
{
const float3 kReferenceEye(2.f, 3.f, 7.f);
const float3 kReferenceTarget(0.f, 0.6f, 0.f);
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, LatticeGPISPass>();
}

LatticeGPISPass::LatticeGPISPass(ref<Device> device, const Properties& props) : RenderPass(device)
{
    mpCamera->setPosition(kReferenceEye);
    mpCamera->setTarget(kReferenceTarget);
    mpCamera->setFocalDistance(1.f);
    // Tungsten's 20-degree field of view is horizontal.
    mpCamera->setFrameWidth(36.f);
    mpCamera->setFocalLength(18.f / std::tan(math::radians(10.f)));
    setProperties(props);
}

void LatticeGPISPass::setProperties(const Properties& props)
{
    const auto known = getProperties();
    for (const auto& [key, value] : props)
        FALCOR_CHECK(known.has(key), "Unknown LatticeGPISPass property '{}'.", key);

    Options options = mOptions;
    props.getTo("sdfFile", options.sdfFile);
    props.getTo("meshFile", options.meshFile);
    props.getTo("clipmapResolution", options.clipmapResolution);
    props.getTo("clipmapLevels", options.clipmapLevels);
    FALCOR_CHECK(!(props.has("sdfFile") && props.has("meshFile") && !options.sdfFile.empty() && !options.meshFile.empty()),
                 "Choose a single geometry source, either meshFile or sdfFile.");
    if (props.has("sdfFile") && !options.sdfFile.empty()) options.meshFile.clear();
    if (props.has("meshFile") && !options.meshFile.empty()) options.sdfFile.clear();
    props.getTo("gridName", options.gridName);
    props.getTo("kernelMode", options.kernelMode);
    props.getTo("correlationLength", options.correlationLength);
    props.getTo("amplitude", options.amplitude);
    props.getTo("correlationLengthEnd", options.correlationLengthEnd);
    props.getTo("amplitudeEnd", options.amplitudeEnd);
    props.getTo("heightRange", options.heightRange);
    props.getTo("latticeSpacingInEll", options.latticeSpacingInEll);
    props.getTo("kernelRadiusInEll", options.kernelRadiusInEll);
    props.getTo("seed", options.seed);

    auto resolve = [](const std::filesystem::path& file)
    {
        auto path = AssetResolver::getDefaultResolver().resolvePath(file);
        FALCOR_CHECK(!path.empty(), "Cannot find geometry file '{}'.", file);
        return path;
    };

    const bool useMesh = !options.meshFile.empty();
    const bool modeChanged = useMesh != bool(mpClipmap);
    auto mesh = mpMesh;
    std::unique_ptr<MeshClipmap> newClipmap;
    std::unique_ptr<SdfField> newField;
    if (useMesh)
    {
        const bool reloadMesh = !mesh || props.has("meshFile");
        if (reloadMesh) mesh = std::make_shared<MeshDistanceField>(resolve(options.meshFile));
        if (reloadMesh || options.clipmapResolution != mOptions.clipmapResolution || options.clipmapLevels != mOptions.clipmapLevels)
            newClipmap = std::make_unique<MeshClipmap>(mpDevice, mesh, options.clipmapResolution, options.clipmapLevels);
    }
    else if (!mpField || (!options.sdfFile.empty() && props.has("sdfFile")) ||
             options.sdfFile != mOptions.sdfFile || options.gridName != mOptions.gridName)
    {
        auto path = options.sdfFile.empty() ? std::filesystem::path() : resolve(options.sdfFile);
        newField = std::make_unique<SdfField>(mpDevice, path, options.gridName);
    }
    const auto bounds = newClipmap ? newClipmap->getBounds() : (newField ? newField->getBounds() : getFieldBounds());
    applyOptions(std::move(options), bounds);
    if (modeChanged) mpTrace = nullptr;
    if (useMesh)
    {
        mpField.reset();
        mpMesh = std::move(mesh);
        if (newClipmap) mpClipmap = std::move(newClipmap);
        if (!mpScene && (modeChanged || props.has("meshFile"))) frameField();
    }
    else
    {
        mpClipmap.reset();
        mpMesh.reset();
        if (newField) mpField = std::move(newField);
    }
    mFileInput = (useMesh ? mOptions.meshFile : mOptions.sdfFile).string();
    mGridInput = mOptions.gridName;
}

void LatticeGPISPass::Options::validate(const AABB& bounds) const
{
    FALCOR_CHECK(
        std::isfinite(correlationLength) && correlationLength > 0.f && std::isfinite(correlationLengthEnd) && correlationLengthEnd > 0.f,
        "LatticeGPISPass requires finite, positive correlation lengths."
    );
    FALCOR_CHECK(
        std::isfinite(amplitude) && amplitude >= 0.f && std::isfinite(amplitudeEnd) && amplitudeEnd >= 0.f,
        "LatticeGPISPass requires finite, nonnegative amplitudes."
    );
    const float heightSpan = heightRange.y - heightRange.x;
    FALCOR_CHECK(std::isfinite(heightSpan) && heightSpan > 0.f, "heightRange must have a finite, positive span.");
    FALCOR_CHECK(std::isfinite(latticeSpacingInEll) && latticeSpacingInEll > 0.f, "Lattice spacing must be finite and positive.");
    FALCOR_CHECK(
        std::isfinite(kernelRadiusInEll) && kernelRadiusInEll > latticeSpacingInEll,
        "The kernel support radius must be finite and greater than the lattice spacing."
    );
    const float endLength = kernelMode == KernelMode::Linear ? correlationLengthEnd : correlationLength;
    const float spacing = latticeSpacingInEll * std::min(correlationLength, endLength);
    FALCOR_CHECK(std::isfinite(spacing) && spacing > 0.f, "The world-space lattice spacing must be finite and positive.");
    const bool varying = hasVaryingKernel();
    const float radiusInCells = std::ceil(kernelRadiusInEll / latticeSpacingInEll);
    FALCOR_CHECK(
        varying || (radiusInCells >= 2.f && radiusInCells <= 32768.f),
        "The stationary kernel support radius must span 2 to 32768 lattice intervals after rounding up."
    );
    const double support = varying ? double(kernelRadiusInEll) * std::max(correlationLength, endLength) : double(radiusInCells) * spacing;
    FALCOR_CHECK(std::isfinite(float(support)), "The world-space kernel support radius exceeds float precision.");
    const auto extent = bounds.extent();
    const double diagonal = std::hypot(double(extent.x), double(extent.y), double(extent.z));
    // Above 2^23 a float index has no fractional bits, so the GPU cannot interpolate adjacent knots.
    FALCOR_CHECK(
        (diagonal + support) / spacing < 8388608.0, "The field and kernel support exceed float lattice index precision at this spacing."
    );
}

void LatticeGPISPass::applyOptions(Options options, const AABB& bounds)
{
    options.validate(bounds);
    updateStationaryWeights(options);
    mOptions = std::move(options);
    mError.clear();
    mReset = true;
}

void LatticeGPISPass::updateStationaryWeights(const Options& options)
{
    // Both modes share the shader layout. Keep a valid binding even when the first field has a varying kernel.
    if (options.hasVaryingKernel() && mpStationaryWeights)
        return;
    const Options defaults;
    const auto& kernel = options.hasVaryingKernel() ? defaults : options;
    const float h = kernel.latticeSpacingInEll;
    const int radius = int(std::ceil(kernel.kernelRadiusInEll / h));
    if (h == mStationarySpacingInEll && radius == mStationaryRadius)
        return;

    // Dimensionless convolution weights and derivatives per lattice interval. The last knot is (0,0).
    std::vector<float2> weights(radius + 1, float2(0.f));
    const float normalization = std::sqrt(h) * 0.893243841738f;
    double valueVariance = 0.0, slopeVariance = 0.0;
    for (int i = 0; i < radius; ++i)
    {
        const float x = float(i) * h;
        const float value = normalization * std::exp(-x * x);
        const float slope = -2.f * x * h * value;
        weights[i] = float2(value, slope);
        const double multiplicity = i == 0 ? 1.0 : 2.0;
        valueVariance += multiplicity * value * value;
        slopeVariance += multiplicity * slope * slope;
    }
    const float2 invVariance(float(1.0 / valueVariance), float(1.0 / slopeVariance));
    FALCOR_CHECK(
        std::isfinite(invVariance.x) && std::isfinite(invVariance.y),
        "The lattice spacing and support cannot resolve the stationary field value and slope at a knot."
    );

    mpStationaryWeights = mpDevice->createStructuredBuffer(
        sizeof(float2), uint32_t(weights.size()), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, weights.data()
    );
    mStationarySpacingInEll = h;
    mStationaryRadius = radius;
    // Preserve the original shader's reference conditioning constants for its default stencil.
    mStationaryInvVariance = h == 0.5f && radius == 6 ? float2(1.000000007f, 4.000002577f) : invVariance;
}

Properties LatticeGPISPass::getProperties() const
{
    Properties props;
    props["sdfFile"] = mOptions.sdfFile;
    props["meshFile"] = mOptions.meshFile;
    props["clipmapResolution"] = mOptions.clipmapResolution;
    props["clipmapLevels"] = mOptions.clipmapLevels;
    props["gridName"] = mOptions.gridName;
    props["kernelMode"] = mOptions.kernelMode;
    props["correlationLength"] = mOptions.correlationLength;
    props["amplitude"] = mOptions.amplitude;
    props["correlationLengthEnd"] = mOptions.correlationLengthEnd;
    props["amplitudeEnd"] = mOptions.amplitudeEnd;
    props["heightRange"] = mOptions.heightRange;
    props["latticeSpacingInEll"] = mOptions.latticeSpacingInEll;
    props["kernelRadiusInEll"] = mOptions.kernelRadiusInEll;
    props["seed"] = mOptions.seed;
    return props;
}

RenderPassReflection LatticeGPISPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput("color", "Linear HDR radiance")
        .texture2D()
        .format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
    return r;
}

void LatticeGPISPass::compile(RenderContext* context, const CompileData& compileData)
{
    mpCamera->setAspectRatio(float(compileData.defaultTexDims.x) / float(compileData.defaultTexDims.y));
    mReset = true;
}

void LatticeGPISPass::setScene(RenderContext* context, const ref<Scene>& scene)
{
    mpScene = scene;
    mController.resetInputState();
    mReset = true;
}

bool LatticeGPISPass::onMouseEvent(const MouseEvent& event)
{
    return !mpScene && mController.onMouseEvent(event);
}

bool LatticeGPISPass::onKeyEvent(const KeyboardEvent& event)
{
    return !mpScene && mController.onKeyEvent(event);
}

AABB LatticeGPISPass::getFieldBounds() const
{
    return mpClipmap ? mpClipmap->getBounds() : mpField->getBounds();
}

void LatticeGPISPass::frameField()
{
    const AABB bounds = mpMesh ? mpMesh->getBounds() : getFieldBounds();
    const auto& camera = mpCamera->getData();
    const float halfExtent = std::min(length(camera.cameraU), length(camera.cameraV)) / length(camera.cameraW);
    const float distance = bounds.radius() * std::sqrt(1.f + halfExtent * halfExtent) / halfExtent;
    mpCamera->setPosition(bounds.center() - normalize(camera.cameraW) * distance);
    mpCamera->setTarget(bounds.center());
    mReset = true;
}

void LatticeGPISPass::execute(RenderContext* context, const RenderData& renderData)
{
    if (!mpScene)
    {
        // Scale navigation to the distance from the mesh so fine detail remains approachable on large models.
        if (mpMesh && (mReset || any(mpCamera->getPosition() != mSpeedSamplePosition)))
        {
            mSpeedSamplePosition = mpCamera->getPosition();
            const float distance = std::abs(mpMesh->signedDistance(mSpeedSamplePosition));
            mController.setCameraSpeed(std::max(2.f * mpClipmap->getFinestSpacing(), distance));
        }
        else if (!mpMesh)
            mController.setCameraSpeed(1.f);
        mReset |= mController.update();
    }
    const auto& camera = (mpScene ? mpScene->getCamera() : mpCamera)->getData();
    if (mpClipmap)
        mpClipmap->update(context, camera.posW);
    if (!mpTrace)
    {
        DefineList defines;
        if (mpClipmap) defines.add("GPIS_MESH_CLIPMAP", "1");
        mpTrace = ComputePass::create(mpDevice, "RenderPasses/LatticeGPIS/LatticeGPIS.cs.slang", "main", defines);
    }
    if (mReset)
    {
        auto& dict = renderData.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mFrameIndex = 0;
        mReset = false;
    }

    auto output = renderData.getTexture("color");
    uint2 dims(output->getWidth(), output->getHeight());
    auto root = mpTrace->getRootVar();
    auto cb = root["CB"];
    cb["gCameraPosition"] = camera.posW;
    cb["gCameraU"] = camera.cameraU;
    cb["gCameraV"] = camera.cameraV;
    cb["gCameraW"] = camera.cameraW;

    cb["gFrameDim"] = dims;
    cb["gSeed"] = mOptions.seed + mFrameIndex;
    cb["gKernelMode"] = uint32_t(mOptions.kernelMode);
    cb["gCorrelationLength"] = mOptions.correlationLength;
    cb["gCorrelationLengthEnd"] = mOptions.correlationLengthEnd;
    cb["gAmplitude"] = mOptions.amplitude;
    cb["gAmplitudeEnd"] = mOptions.amplitudeEnd;
    cb["gHeightRange"] = mOptions.heightRange;
    cb["gLatticeSpacingInEll"] = mOptions.latticeSpacingInEll;
    cb["gKernelRadiusInEll"] = mOptions.kernelRadiusInEll;
    cb["gStationaryRadius"] = mStationaryRadius;
    cb["gStationaryInvVariance"] = mStationaryInvVariance;
    root["gStationaryWeights"] = mpStationaryWeights;
    if (mpClipmap) mpClipmap->bind(root["gField"]);
    else mpField->bind(root["gField"]);
    root["gOutput"] = output;
    mpTrace->execute(context, uint3(dims, 1));
    ++mFrameIndex;
}

void LatticeGPISPass::renderUI(Gui::Widgets& widget)
{
    widget.textbox("Geometry file", mFileInput);
    widget.tooltip("Choose a closed triangle mesh or an OpenVDB field. Leave empty for the reference shaderball.");
    bool load = widget.button("Browse...");
    if (load)
    {
        std::filesystem::path file;
        load = openFileDialog({{"obj", "Wavefront mesh"}, {"ply", "PLY mesh"}, {"gltf", "glTF mesh"}, {"glb", "Binary glTF mesh"},
                               {"vdb", "OpenVDB field"}}, file);
        if (load) mFileInput = file.string();
    }
    load |= widget.button("Load geometry", true);
    const bool meshInput = !mFileInput.empty() && !hasExtension(mFileInput, "vdb");
    if (!meshInput) widget.textbox("Grid name", mGridInput);
    if (load)
    {
        Properties props;
        props["meshFile"] = meshInput ? mFileInput : "";
        props["sdfFile"] = meshInput ? "" : mFileInput;
        props["gridName"] = mGridInput;
        try
        {
            setProperties(props);
        }
        catch (const std::exception& e)
        {
            mError = e.what();
        }
    }

    if (mpClipmap)
    {
        uint32_t levels = mOptions.clipmapLevels;
        uint32_t resolution = mOptions.clipmapResolution;
        const Gui::DropdownList resolutions = {{9, "9"}, {17, "17"}, {33, "33"}, {65, "65"}, {129, "129"}};
        bool changed = widget.dropdown("Clipmap nodes per axis", resolutions, resolution);
        changed |= widget.var("Clipmap levels", levels, 1u, 12u);
        if (changed)
        {
            Properties props;
            props["clipmapResolution"] = resolution;
            props["clipmapLevels"] = levels;
            try { setProperties(props); }
            catch (const std::exception& e) { mError = e.what(); }
        }
        const uint32_t active = mpClipmap->getActiveLevelCount();
        widget.text(fmt::format("Active clipmap levels: {} / {}", active, mOptions.clipmapLevels));
        widget.text(fmt::format("Finest active voxel: {:.5g} world units",
                                std::ldexp(mpClipmap->getFinestSpacing(), int(mOptions.clipmapLevels - active))));
        widget.text(fmt::format("SDF nodes updated this frame: {}", mpClipmap->getUpdatedVoxelCount()));
    }

    if (!mpScene)
    {
        if (widget.button("Frame SDF")) frameField();
        if (widget.button("Shaderball camera", true))
        {
            mpCamera->setPosition(kReferenceEye);
            mpCamera->setTarget(kReferenceTarget);
            mpCamera->setUpVector(float3(0.f, 1.f, 0.f));
            mReset = true;
        }
    }

    Options options = mOptions;
    bool changed = widget.dropdown("Kernel", options.kernelMode);
    changed |= widget.var("Correlation length", options.correlationLength, 0.f);
    changed |= widget.var("Amplitude", options.amplitude, 0.f);
    if (options.kernelMode == KernelMode::Linear)
    {
        changed |= widget.var("End correlation length", options.correlationLengthEnd, 0.f);
        changed |= widget.var("End amplitude", options.amplitudeEnd, 0.f);
        changed |= widget.var("Height range (Y)", options.heightRange);
    }
    changed |= widget.var("Lattice spacing (in ell)", options.latticeSpacingInEll, 0.f, std::numeric_limits<float>::max(), 0.01f);
    widget.tooltip("Spacing between noise knots, in units of the smallest correlation length.");
    changed |= widget.var("Kernel support radius (in ell)", options.kernelRadiusInEll, 0.f, std::numeric_limits<float>::max(), 0.05f);
    widget.tooltip(
        "One-sided radius in local correlation lengths. The stationary kernel rounds its radius up to a whole number of intervals."
    );
    changed |= widget.var("Seed", options.seed);
    if (changed)
    {
        try
        {
            applyOptions(std::move(options), getFieldBounds());
        }
        catch (const std::exception& e)
        {
            mError = e.what();
        }
    }
    if (!mError.empty())
        widget.textWrapped(mError);

    const bool varying = mOptions.hasVaryingKernel();
    const float ellMin = varying ? std::min(mOptions.correlationLength, mOptions.correlationLengthEnd) : mOptions.correlationLength;
    const float ellMax = varying ? std::max(mOptions.correlationLength, mOptions.correlationLengthEnd) : mOptions.correlationLength;
    const double spacing = double(mOptions.latticeSpacingInEll) * ellMin;
    widget.text(fmt::format("World lattice spacing: {:.5g}", spacing));
    if (varying && ellMin != ellMax)
        widget.text(fmt::format(
            "World support width: {:.5g} to {:.5g}", 2.0 * mOptions.kernelRadiusInEll * ellMin, 2.0 * mOptions.kernelRadiusInEll * ellMax
        ));
    else
        widget.text(fmt::format(
            "World support width: {:.5g}", varying ? 2.0 * mOptions.kernelRadiusInEll * ellMin : 2.0 * mStationaryRadius * spacing
        ));
}
