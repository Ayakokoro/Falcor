#include "LatticeGPISPass.h"
#include "BakedGPISPass.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/Image/Bitmap.h"

namespace
{
const float3 kReferenceEye(2.f, 3.f, 7.f);
const float3 kReferenceTarget(0.f, 0.6f, 0.f);
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, LatticeGPISPass>();
    registry.registerClass<RenderPass, BakedGPISPass>();
    ScriptBindings::registerBinding([](pybind11::module& m) {
        m.def("bake_gpis_density", &DensityClipmap::bakeGPIS, pybind11::arg("source"), pybind11::arg("grid"),
              pybind11::arg("output"), pybind11::arg("ell"), pybind11::arg("sigma"),
              pybind11::arg("extinction_scale"), pybind11::arg("world_scale"), pybind11::arg("samples_per_voxel") = 1);
    });
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
    if (options.meshFile != mOptions.meshFile) options.baseColorTexture.clear();
    props.getTo("baseColorTexture", options.baseColorTexture);
    FALCOR_CHECK(options.baseColorTexture.empty() || !options.meshFile.empty(), "A base-color texture requires a mesh file.");
    props.getTo("gridName", options.gridName);
    props.getTo("kernelMode", options.kernelMode);
    props.getTo("correlationLength", options.correlationLength);
    props.getTo("amplitude", options.amplitude);
    props.getTo("correlationLengthEnd", options.correlationLengthEnd);
    props.getTo("amplitudeEnd", options.amplitudeEnd);
    props.getTo("heightRange", options.heightRange);
    props.getTo("latticeSpacingInEll", options.latticeSpacingInEll);
    props.getTo("kernelRadiusInEll", options.kernelRadiusInEll);
    props.getTo("surfaceEnabled", options.surfaceEnabled);
    props.getTo("cloudEnabled", options.cloudEnabled);
    props.getTo("cloudCenter", options.cloudCenter);
    props.getTo("cloudRadii", options.cloudRadii);
    props.getTo("cloudCorrelationLength", options.cloudCorrelationLength);
    props.getTo("cloudAmplitude", options.cloudAmplitude);
    props.getTo("cloudMeanOffset", options.cloudMeanOffset);
    props.getTo("cloudColor", options.cloudColor);
    props.getTo("cloudDensityFile", options.cloudDensityFile);
    props.getTo("cloudDensityGrid", options.cloudDensityGrid);
    props.getTo("densityUseClipmap", options.densityUseClipmap);
    props.getTo("densityClipmapResolution", options.densityClipmapResolution);
    props.getTo("densityClipmapLevels", options.densityClipmapLevels);
    props.getTo("cloudExtinctionScale", options.cloudExtinctionScale);
    props.getTo("cloudDensityWorldScale", options.cloudDensityWorldScale);
    props.getTo("cloudDensityTranslation", options.cloudDensityTranslation);
    props.getTo("seed", options.seed);

    auto resolve = [](const std::filesystem::path& file)
    {
        auto path = AssetResolver::getDefaultResolver().resolvePath(file);
        FALCOR_CHECK(!path.empty(), "Cannot find asset '{}'.", file);
        return path;
    };

    const bool useMesh = !options.meshFile.empty();
    const bool modeChanged = useMesh != bool(mpClipmap);
    auto mesh = mpMesh;
    auto texture = mpBaseColor;
    auto surfaceUV = mSurfaceUV;
    std::unique_ptr<MeshClipmap> newClipmap;
    std::unique_ptr<SdfField> newField;
    if (useMesh)
    {
        const bool explicitTexture = !options.baseColorTexture.empty();
        const bool reloadMesh = !mesh || props.has("meshFile") || (explicitTexture && !mesh->hasTexCoords());
        if (reloadMesh) mesh = std::make_shared<MeshDistanceField>(resolve(options.meshFile), explicitTexture);
        if (reloadMesh || props.has("baseColorTexture"))
        {
            if (reloadMesh) surfaceUV = {};
            auto path = mesh->getBaseColorTexturePath();
            if (explicitTexture)
            {
                FALCOR_CHECK(mesh->hasTexCoords(), "The mesh '{}' needs UV coordinates to apply a base-color texture.", options.meshFile);
                path = resolve(resolve(options.meshFile).parent_path() / options.baseColorTexture);
            }
            texture = nullptr;
            if (!path.empty() && mesh->hasTexCoords())
            {
                texture = Texture::createFromFile(mpDevice, path, true, true);
                if (!texture || texture->getType() != Resource::Type::Texture2D || texture->getArraySize() != 1)
                {
                    FALCOR_CHECK(!explicitTexture, "Could not load '{}' as a 2D base-color texture.", path);
                    logWarning("Lattice GPIS could not load mesh base-color texture '{}'.", path);
                    texture = nullptr;
                }
            }
            if (texture)
            {
                if (!surfaceUV.nodes) surfaceUV = mesh->createSurfaceUVBuffers(mpDevice);
                if (!mpMeshSampler) mpMeshSampler = mpDevice->createSampler(Sampler::Desc());
            }
            else surfaceUV = {};
        }
        if (reloadMesh || options.clipmapResolution != mOptions.clipmapResolution || options.clipmapLevels != mOptions.clipmapLevels)
            newClipmap = std::make_unique<MeshClipmap>(mpDevice, mesh, options.clipmapResolution, options.clipmapLevels);
    }
    else if (!mpField || (!options.sdfFile.empty() && props.has("sdfFile")) ||
             options.sdfFile != mOptions.sdfFile || options.gridName != mOptions.gridName)
    {
        auto path = options.sdfFile.empty() ? std::filesystem::path() : resolve(options.sdfFile);
        newField = std::make_unique<SdfField>(mpDevice, path, options.gridName);
    }
    const bool useDensity = !options.cloudDensityFile.empty();
    std::unique_ptr<DensityClipmap> newDensity;
    if (useDensity && (!mpDensityClipmap || props.has("cloudDensityFile") || options.cloudDensityGrid != mOptions.cloudDensityGrid ||
                       options.densityUseClipmap != mOptions.densityUseClipmap ||
                       (options.densityUseClipmap && (options.densityClipmapLevels != mOptions.densityClipmapLevels ||
                                                     options.densityClipmapResolution != mOptions.densityClipmapResolution))))
        newDensity = std::make_unique<DensityClipmap>(mpDevice, resolve(options.cloudDensityFile), options.cloudDensityGrid,
                                                       options.densityClipmapResolution, options.densityClipmapLevels, options.densityUseClipmap);
    const bool frameDensity = newDensity && options.cloudEnabled && (!mpDensityClipmap || props.has("cloudDensityFile"));
    const auto bounds = newClipmap ? newClipmap->getBounds() : (newField ? newField->getBounds() : getFieldBounds());
    applyOptions(std::move(options), bounds, useDensity ? (newDensity ? newDensity.get() : mpDensityClipmap.get()) : nullptr);
    if (newDensity) mpDensityClipmap = std::move(newDensity);
    if (!useDensity) mpDensityClipmap.reset();
    if (!useMesh) { texture = nullptr; surfaceUV = {}; }
    if (modeChanged || bool(texture) != bool(mpBaseColor)) mpTrace = nullptr;
    mpBaseColor = std::move(texture);
    mSurfaceUV = std::move(surfaceUV);
    if (useMesh)
    {
        mpField.reset();
        mpMesh = std::move(mesh);
        if (newClipmap) mpClipmap = std::move(newClipmap);
    }
    else
    {
        mpClipmap.reset();
        mpMesh.reset();
        if (newField) mpField = std::move(newField);
    }
    if (!mpScene && (frameDensity || (useMesh && (modeChanged || props.has("meshFile"))))) frameField();
    mFileInput = (useMesh ? mOptions.meshFile : mOptions.sdfFile).string();
    mGridInput = mOptions.gridName;
    mTextureInput = mOptions.baseColorTexture.string();
    mDensityFileInput = mOptions.cloudDensityFile.string();
    mDensityGridInput = mOptions.cloudDensityGrid;
}

AABB LatticeGPISPass::Options::cloudBounds(const DensityClipmap* density) const
{
    if (!density)
        return AABB(cloudCenter - cloudRadii, cloudCenter + cloudRadii);
    const auto b = density->getBounds();
    return AABB(b.minPoint * cloudDensityWorldScale + cloudDensityTranslation, b.maxPoint * cloudDensityWorldScale + cloudDensityTranslation);
}

void LatticeGPISPass::Options::validate(const AABB& bounds, const DensityClipmap* density) const
{
    FALCOR_CHECK(std::isfinite(cloudExtinctionScale) && cloudExtinctionScale >= 0.f, "Cloud extinction scale must be finite and nonnegative.");
    FALCOR_CHECK(std::isfinite(cloudDensityWorldScale) && cloudDensityWorldScale > 0.f && std::isfinite(1.f / cloudDensityWorldScale),
                 "Cloud density world scale must be finite and positive.");
    FALCOR_CHECK(densityClipmapResolution >= 9 && densityClipmapResolution <= 129 &&
                 ((densityClipmapResolution - 1) & (densityClipmapResolution - 2)) == 0 &&
                 densityClipmapLevels >= 1 && densityClipmapLevels <= 12,
                 "Density clipmaps require 9, 17, 33, 65 or 129 nodes per axis and 1 to 12 levels.");
    FALCOR_CHECK(!cloudDensityGrid.empty(), "Cloud density grid name must not be empty.");
    const AABB cloud = cloudBounds(density);
    for (int axis = 0; axis < 3; ++axis)
    {
        const float lo = cloudCenter[axis] - cloudRadii[axis];
        const float hi = cloudCenter[axis] + cloudRadii[axis];
        FALCOR_CHECK(std::isfinite(lo) && std::isfinite(hi) && lo < hi,
                     "The cloud center and radii must define positive, finite world-space intervals.");
        FALCOR_CHECK(cloudColor[axis] >= 0.f && cloudColor[axis] <= 1.f, "Cloud albedo must be in [0, 1].");
        FALCOR_CHECK(std::isfinite(cloudDensityTranslation[axis]) && std::isfinite(cloud.minPoint[axis]) &&
                     std::isfinite(cloud.maxPoint[axis]) && cloud.minPoint[axis] < cloud.maxPoint[axis],
                     "Cloud density bounds must be finite and representable after placement.");
    }
    FALCOR_CHECK(std::isfinite(cloudCorrelationLength) && cloudCorrelationLength > 0.f &&
                 std::isfinite(cloudAmplitude) && cloudAmplitude >= 0.f && std::isfinite(cloudMeanOffset),
                 "Cloud correlation length must be finite and positive, amplitude finite and nonnegative, and mean offset finite.");
    if (density)
    {
        const float factor = 2.506628275f * cloudCorrelationLength * cloudExtinctionScale;
        FALCOR_CHECK(std::isfinite(factor) && std::isfinite(factor * density->getMaxDensity()),
                     "The product of cloud correlation length, extinction scale and maximum density exceeds float precision.");
    }
    const auto cloudExtent = cloud.extent();
    const double cloudDiameter = density ? std::hypot(double(cloudExtent.x), double(cloudExtent.y), double(cloudExtent.z))
                                         : 2.0 * std::max({cloudRadii.x, cloudRadii.y, cloudRadii.z});
    const float cloudSpacing = 0.5f * cloudCorrelationLength;
    FALCOR_CHECK(std::isfinite(float(cloudDiameter)) &&
                 (cloudAmplitude == 0.f || (cloudSpacing > 0.f && cloudDiameter / cloudSpacing + 6.0 < 8388608.0)),
                 "The cloud diameter must be finite and its lattice knots representable at this correlation length.");
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

void LatticeGPISPass::applyOptions(Options options, const AABB& bounds, const DensityClipmap* density)
{
    options.validate(bounds, density);
    updateStationaryWeights(options);
    if (options.usesDualFieldShader() != mOptions.usesDualFieldShader() || options.usesDensityShader() != mOptions.usesDensityShader())
        mpTrace = nullptr;
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
    props["baseColorTexture"] = mOptions.baseColorTexture;
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
    props["surfaceEnabled"] = mOptions.surfaceEnabled;
    props["cloudEnabled"] = mOptions.cloudEnabled;
    props["cloudCenter"] = mOptions.cloudCenter;
    props["cloudRadii"] = mOptions.cloudRadii;
    props["cloudCorrelationLength"] = mOptions.cloudCorrelationLength;
    props["cloudAmplitude"] = mOptions.cloudAmplitude;
    props["cloudMeanOffset"] = mOptions.cloudMeanOffset;
    props["cloudColor"] = mOptions.cloudColor;
    props["cloudDensityFile"] = mOptions.cloudDensityFile;
    props["cloudDensityGrid"] = mOptions.cloudDensityGrid;
    props["densityUseClipmap"] = mOptions.densityUseClipmap;
    props["densityClipmapResolution"] = mOptions.densityClipmapResolution;
    props["densityClipmapLevels"] = mOptions.densityClipmapLevels;
    props["cloudExtinctionScale"] = mOptions.cloudExtinctionScale;
    props["cloudDensityWorldScale"] = mOptions.cloudDensityWorldScale;
    props["cloudDensityTranslation"] = mOptions.cloudDensityTranslation;
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
    AABB bounds = mpMesh ? mpMesh->getBounds() : getFieldBounds();
    if (mOptions.cloudEnabled)
    {
        const AABB cloudBounds = mOptions.cloudBounds(mpDensityClipmap.get());
        if (mOptions.surfaceEnabled) bounds.include(cloudBounds);
        else bounds = cloudBounds;
    }
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
    if (mpClipmap && mOptions.surfaceEnabled)
        mpClipmap->update(context, camera.posW);
    if (mOptions.usesDensityShader())
        mpDensityClipmap->update(context, (camera.posW - mOptions.cloudDensityTranslation) / mOptions.cloudDensityWorldScale);
    if (!mpTrace)
    {
        DefineList defines;
        if (mpClipmap) defines.add("GPIS_MESH_CLIPMAP", "1");
        if (mpBaseColor) defines.add("GPIS_TEXTURED_MESH", "1");
        if (mOptions.usesDualFieldShader()) defines.add("GPIS_SECOND_FIELD", "1");
        if (mOptions.usesDensityShader()) defines.add("GPIS_DENSITY_CLIPMAP", "1");
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
    if (mOptions.usesDualFieldShader())
    {
        cb["gSurfaceEnabled"] = uint32_t(mOptions.surfaceEnabled);
        cb["gCloudEnabled"] = uint32_t(mOptions.cloudEnabled);
        cb["gCloudCenter"] = mOptions.cloudCenter;
        cb["gCloudRadii"] = mOptions.cloudRadii;
        cb["gCloudCorrelationLength"] = mOptions.cloudCorrelationLength;
        cb["gCloudAmplitude"] = mOptions.cloudAmplitude;
        cb["gCloudMeanOffset"] = mOptions.cloudMeanOffset;
        cb["gCloudColor"] = mOptions.cloudColor;
    }
    if (mOptions.usesDensityShader())
    {
        cb["gDensityExtinctionScale"] = mOptions.cloudExtinctionScale;
        cb["gDensityTranslation"] = mOptions.cloudDensityTranslation;
        cb["gDensityWorldScale"] = mOptions.cloudDensityWorldScale;
        mpDensityClipmap->bind(root["gDensity"]);
    }
    root["gStationaryWeights"] = mpStationaryWeights;
    if (mpClipmap) mpClipmap->bind(root["gField"]);
    else mpField->bind(root["gField"]);
    if (mpBaseColor)
    {
        mSurfaceUV.bind(root);
        root["gMeshBaseColor"] = mpBaseColor;
        root["gMeshSampler"] = mpMeshSampler;
    }
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
        widget.textbox("Base-color texture", mTextureInput);
        widget.tooltip("An empty override uses the mesh material's shared base-color map. Relative paths use the mesh folder.");
        bool applyTexture = widget.button("Browse texture...");
        if (applyTexture)
        {
            std::filesystem::path path;
            applyTexture = openFileDialog(Bitmap::getFileDialogFilters(), path);
            if (applyTexture) mTextureInput = path.string();
        }
        applyTexture |= widget.button("Apply texture", true);
        if (applyTexture)
        {
            Properties props;
            props["baseColorTexture"] = mTextureInput;
            try { setProperties(props); }
            catch (const std::exception& e) { mError = e.what(); }
        }
        if (mpBaseColor) widget.text(fmt::format("Using texture: {}", mpBaseColor->getSourcePath().filename().string()));
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
        widget.text(fmt::format("SDF nodes updated this frame: {}", mOptions.surfaceEnabled ? mpClipmap->getUpdatedVoxelCount() : 0));
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

    if (mOptions.cloudEnabled) renderDensitySourceUI(widget);
    Options options = mOptions;
    bool changed = widget.checkbox("Render surface", options.surfaceEnabled);
    changed |= widget.checkbox("Render cloudlike GPIS", options.cloudEnabled);
    if (options.cloudEnabled)
    {
        if (auto cloud = widget.group("Cloudlike GPIS parameters"))
        {
            if (mpDensityClipmap)
            {
                changed |= cloud.var("Density world scale", options.cloudDensityWorldScale, 0.f);
                changed |= cloud.var("Density translation", options.cloudDensityTranslation);
                changed |= cloud.var("Extinction per unit density", options.cloudExtinctionScale, 0.f);
            }
            else
            {
                changed |= cloud.var("Center", options.cloudCenter);
                changed |= cloud.var("Radii", options.cloudRadii, 0.f);
                changed |= cloud.var("Mean at center", options.cloudMeanOffset);
                cloud.tooltip("Increasing the mean makes random zero crossings rarer at the same amplitude.");
            }
            changed |= cloud.var("Correlation length", options.cloudCorrelationLength, 0.f);
            changed |= cloud.var("Amplitude", options.cloudAmplitude, 0.f);
            if (mpDensityClipmap) cloud.tooltip("For a density grid the mean scales with amplitude. Positive amplitudes share the same zero crossings; zero disables cloud hits.");
            changed |= cloud.rgbColor("Albedo", options.cloudColor);
        }
    }
    changed |= widget.dropdown("Kernel", options.kernelMode);
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
            applyOptions(std::move(options), getFieldBounds(), mpDensityClipmap.get());
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

void LatticeGPISPass::renderDensitySourceUI(Gui::Widgets& widget)
{
    auto group = widget.group("Cloud density source");
    if (!group) return;
    group.textbox("Density VDB", mDensityFileInput);
    group.tooltip("Use a fog density grid, or clear the file to use the analytic ellipsoid.");
    group.textbox("Density grid", mDensityGridInput);
    bool load = group.button("Browse density...");
    if (load)
    {
        std::filesystem::path path;
        load = openFileDialog({{"vdb", "OpenVDB density grid"}}, path);
        if (load) mDensityFileInput = path.string();
    }
    load |= group.button("Apply density", true);
    Properties props;
    if (load)
    {
        props["cloudDensityFile"] = mDensityFileInput;
        props["cloudDensityGrid"] = mDensityGridInput;
    }
    if (mpDensityClipmap)
    {
        bool useClipmap = mOptions.densityUseClipmap;
        if (group.checkbox("Use density clipmap", useClipmap)) props["densityUseClipmap"] = useClipmap;
    }
    if (mpDensityClipmap && mOptions.densityUseClipmap)
    {
        const Gui::DropdownList sizes = {{9, "9"}, {17, "17"}, {33, "33"}, {65, "65"}, {129, "129"}};
        uint32_t resolution = mOptions.densityClipmapResolution;
        uint32_t levels = mOptions.densityClipmapLevels;
        if (group.dropdown("Density nodes per axis", sizes, resolution)) props["densityClipmapResolution"] = resolution;
        if (group.var("Density levels", levels, 1u, 12u)) props["densityClipmapLevels"] = levels;
        group.text(fmt::format("Active density levels: {}   Nodes updated: {}", mpDensityClipmap->getActiveLevelCount(),
                               mpDensityClipmap->getUpdatedVoxelCount()));
    }
    if (!props.empty())
    {
        try { setProperties(props); }
        catch (const std::exception& e) { mError = e.what(); }
    }
}
