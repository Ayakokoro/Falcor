#include "VolumeReferencePass.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include <cmath>

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, VolumeReferencePass>();
}

VolumeReferencePass::VolumeReferencePass(ref<Device> device, const Properties& props) : RenderPass(device)
{
    mpFallbackCamera->setPosition(float3(2.f, 3.f, 7.f));
    mpFallbackCamera->setTarget(float3(0.f, 0.6f, 0.f));
    mpFallbackCamera->setFocalDistance(1.f);
    mpFallbackCamera->setFrameWidth(36.f);
    mpFallbackCamera->setFocalLength(18.f / std::tan(math::radians(10.f)));
    setProperties(props);
}

Properties VolumeReferencePass::getProperties() const
{
    Properties props;
    props["cloudDensityFile"] = mOptions.file;
    props["cloudDensityGrid"] = mOptions.grid;
    props["cloudExtinctionScale"] = mOptions.extinction;
    props["cloudDensityWorldScale"] = mOptions.worldScale;
    props["cloudDensityTranslation"] = mOptions.translation;
    props["densityUseClipmap"] = mOptions.useClipmap;
    props["densityClipmapResolution"] = mOptions.resolution;
    props["densityClipmapLevels"] = mOptions.levels;
    props["seed"] = mOptions.seed;
    return props;
}

void VolumeReferencePass::setProperties(const Properties& props)
{
    const auto known = getProperties();
    for (const auto& [key, value] : props) FALCOR_CHECK(known.has(key), "Unknown VolumeReferencePass property '{}'.", key);
    Options next = mOptions;
    props.getTo("cloudDensityFile", next.file);
    props.getTo("cloudDensityGrid", next.grid);
    props.getTo("cloudExtinctionScale", next.extinction);
    props.getTo("cloudDensityWorldScale", next.worldScale);
    props.getTo("cloudDensityTranslation", next.translation);
    props.getTo("densityUseClipmap", next.useClipmap);
    props.getTo("densityClipmapResolution", next.resolution);
    props.getTo("densityClipmapLevels", next.levels);
    props.getTo("seed", next.seed);
    FALCOR_CHECK(!next.grid.empty(), "The density grid name must not be empty.");
    FALCOR_CHECK(std::isfinite(next.extinction) && next.extinction >= 0.f, "The extinction scale must be finite and nonnegative.");
    FALCOR_CHECK(std::isfinite(next.worldScale) && next.worldScale > 0.f && std::isfinite(1.f / next.worldScale),
                 "The density world scale must be finite and positive.");
    FALCOR_CHECK(next.resolution >= 9 && next.resolution <= 129 && ((next.resolution - 1) & (next.resolution - 2)) == 0 &&
                 next.levels >= 1 && next.levels <= 12, "Invalid density clipmap size.");
    const bool newGrid = next.file != mOptions.file || props.has("cloudDensityFile") || next.grid != mOptions.grid ||
                         next.useClipmap != mOptions.useClipmap ||
                         (next.useClipmap && (next.resolution != mOptions.resolution || next.levels != mOptions.levels));
    std::unique_ptr<DensityClipmap> replacement;
    if (newGrid && !next.file.empty())
    {
        const auto path = AssetResolver::getDefaultResolver().resolvePath(next.file);
        FALCOR_CHECK(!path.empty(), "Cannot find density asset '{}'.", next.file);
        replacement = std::make_unique<DensityClipmap>(mpDevice, path, next.grid, next.resolution, next.levels, next.useClipmap);
    }
    const auto* volume = newGrid ? replacement.get() : mpDensity.get();
    for (int axis = 0; axis < 3; ++axis)
    {
        FALCOR_CHECK(std::isfinite(next.translation[axis]), "The density translation must be finite.");
        if (volume)
        {
            const AABB bounds = volume->getBounds();
            const float lo = bounds.minPoint[axis] * next.worldScale + next.translation[axis];
            const float hi = bounds.maxPoint[axis] * next.worldScale + next.translation[axis];
            FALCOR_CHECK(std::isfinite(lo) && std::isfinite(hi) && lo < hi, "Density bounds exceed world-coordinate precision.");
        }
    }
    if (newGrid) mpDensity = std::move(replacement);
    mOptions = std::move(next);
    if (!mpScene) frameVolume();
    mReset = true;
    mError.clear();
}

void VolumeReferencePass::frameVolume()
{
    if (!mpDensity) return;
    const AABB bounds = mpDensity->getBounds();
    const auto& camera = mpFallbackCamera->getData();
    const float tangent = std::min(length(camera.cameraU), length(camera.cameraV)) / length(camera.cameraW);
    const float distance = bounds.radius() * mOptions.worldScale * std::sqrt(1.f + tangent * tangent) / tangent;
    const float3 center = bounds.center() * mOptions.worldScale + mOptions.translation;
    mpFallbackCamera->setPosition(center - normalize(camera.cameraW) * distance);
    mpFallbackCamera->setTarget(center);
}

RenderPassReflection VolumeReferencePass::reflect(const CompileData&)
{
    RenderPassReflection r;
    r.addOutput("transmittance", "Linear unscattered transmittance, RGB = T, alpha = 1")
        .flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    r.addOutput("radiance", "Linear single-scattered radiance and attenuated background")
        .flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    return r;
}

void VolumeReferencePass::compile(RenderContext*, const CompileData& data)
{
    mpFallbackCamera->setAspectRatio(float(data.defaultTexDims.x) / float(data.defaultTexDims.y));
    mpIntegrate.reset();
    mReset = true;
}

void VolumeReferencePass::setScene(RenderContext*, const ref<Scene>& scene)
{
    mpScene = scene;
    mReset = true;
}

void VolumeReferencePass::execute(RenderContext* context, const RenderData& data)
{
    if (mReset)
    {
        auto& dict = data.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mFrameIndex = 0;
        mReset = false;
    }
    const auto transmittance = data.getTexture("transmittance");
    const auto radiance = data.getTexture("radiance");
    if (!transmittance && !radiance) return;
    if (!mpDensity)
    {
        if (transmittance) context->clearUAV(transmittance->getUAV().get(), float4(1.f));
        if (radiance) context->clearUAV(radiance->getUAV().get(), float4(.22f, .30f, .42f, 1.f));
        return;
    }
    if (!mpIntegrate)
    {
        DefineList defines;
        defines.add("WRITE_TRANSMITTANCE", transmittance ? "1" : "0");
        defines.add("WRITE_RADIANCE", radiance ? "1" : "0");
        mpIntegrate = ComputePass::create(mpDevice, "RenderPasses/VolumeReferencePass/VolumeReference.cs.slang", "main", defines);
    }
    const auto& camera = (mpScene ? mpScene->getCamera() : mpFallbackCamera)->getData();
    mpDensity->update(context, (camera.posW - mOptions.translation) / mOptions.worldScale);
    const auto& output = transmittance ? transmittance : radiance;
    const uint2 dims(output->getWidth(), output->getHeight());
    auto vars = mpIntegrate->getRootVar();
    vars["gCameraPosition"] = camera.posW;
    vars["gCameraU"] = camera.cameraU;
    vars["gCameraV"] = camera.cameraV;
    vars["gCameraW"] = camera.cameraW;
    vars["gFrameDim"] = dims;
    vars["gSeed"] = mOptions.seed + mFrameIndex++;
    vars["gDensityTranslation"] = mOptions.translation;
    vars["gDensityWorldScale"] = mOptions.worldScale;
    vars["gExtinctionScale"] = mOptions.extinction;
    if (transmittance) vars["gTransmittance"] = transmittance;
    if (radiance) vars["gRadiance"] = radiance;
    mpDensity->bind(vars["gDensity"]);
    mpIntegrate->execute(context, uint3(dims, 1));
}

void VolumeReferencePass::renderUI(Gui::Widgets& widget)
{
    Properties props;
    if (widget.button("Load density VDB..."))
    {
        std::filesystem::path path;
        if (openFileDialog({{"vdb", "OpenVDB density grid"}}, path)) props["cloudDensityFile"] = path;
    }
    if (mpDensity) widget.text(mOptions.file.filename().string());
    float extinction = mOptions.extinction;
    bool clipmap = mOptions.useClipmap;
    if (widget.var("Extinction per unit density", extinction, 0.f)) props["cloudExtinctionScale"] = extinction;
    if (widget.checkbox("Use density clipmap", clipmap)) props["densityUseClipmap"] = clipmap;
    if (!props.empty())
    {
        try { setProperties(props); }
        catch (const std::exception& e) { mError = e.what(); }
    }
    if (!mError.empty()) widget.textWrapped(mError);
}
