#include "BakedGPISPass.h"
#include "Core/AssetResolver.h"
#include "RenderGraph/RenderPassStandardFlags.h"

using namespace Falcor;

BakedGPISPass::BakedGPISPass(ref<Device> device, const Properties& props) : RenderPass(device)
{
    mpFallbackCamera->setPosition(float3(2.f, 3.f, 7.f));
    mpFallbackCamera->setTarget(float3(0.f, 0.6f, 0.f));
    mpFallbackCamera->setFocalDistance(1.f);
    mpFallbackCamera->setFrameWidth(36.f);
    mpFallbackCamera->setFocalLength(18.f / std::tan(math::radians(10.f)));
    setProperties(props);
}

Properties BakedGPISPass::getProperties() const
{
    Properties props;
    props["cloudBakedFile"] = mFile;
    props["cloudDensityTranslation"] = mTranslation;
    props["seed"] = mSeed;
    return props;
}

void BakedGPISPass::setProperties(const Properties& props)
{
    const auto known = getProperties();
    for (const auto& [key, value] : props)
        FALCOR_CHECK(known.has(key), "Unknown BakedGPISPass property '{}'.", key);
    auto file = mFile;
    auto placement = mTranslation;
    auto seed = mSeed;
    props.getTo("cloudBakedFile", file);
    props.getTo("cloudDensityTranslation", placement);
    props.getTo("seed", seed);
    FALCOR_CHECK(all(isfinite(placement)), "GPIS translation must be finite.");
    std::unique_ptr<DensityClipmap> replacement;
    const bool reload = props.has("cloudBakedFile");
    if (reload && !file.empty())
    {
        const auto path = AssetResolver::getDefaultResolver().resolvePath(file);
        FALCOR_CHECK(!path.empty(), "Cannot find baked GPIS '{}'.", file);
        replacement = std::make_unique<DensityClipmap>(mpDevice, path);
    }
    auto* field = reload ? replacement.get() : mpMean.get();
    if (field)
    {
        const AABB source = field->getBounds();
        const float3 lo = source.minPoint * field->getBakedWorldScale() + placement;
        const float3 hi = source.maxPoint * field->getBakedWorldScale() + placement;
        FALCOR_CHECK(all(isfinite(lo)) && all(isfinite(hi)) && all(lo < hi),
                     "Baked GPIS cannot be represented at this world-space placement.");
        FALCOR_CHECK(double(length(hi - lo)) / (0.5 * field->getBakedCorrelationLength()) + 6.0 < 8388608.0,
                     "The baked field extent exceeds float32 lattice coordinate precision.");
        if (!mpScene)
        {
            const auto& camera = mpFallbackCamera->getData();
            const float tangent = std::min(length(camera.cameraU), length(camera.cameraV)) / length(camera.cameraW);
            const float3 center = (lo + hi) * 0.5f;
            const float distance = .5f * length(hi-lo) * std::sqrt(1.f + tangent*tangent) / tangent;
            mpFallbackCamera->setPosition(center - normalize(camera.cameraW) * distance);
            mpFallbackCamera->setTarget(center);
        }
    }
    if (reload) mpMean = std::move(replacement);
    mFile = file;
    mTranslation = placement;
    mSeed = seed;
    mReset = true;
    mError.clear();
}

RenderPassReflection BakedGPISPass::reflect(const CompileData&)
{
    RenderPassReflection r;
    r.addOutput("color", "Single-scattered radiance from baked GPIS flights")
        .flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    r.addOutput("transmittance", "Unoccluded primary rays, average over samples to measure transmittance")
        .flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float)
        .bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    return r;
}

void BakedGPISPass::compile(RenderContext*, const CompileData& data)
{
    mpFallbackCamera->setAspectRatio(float(data.defaultTexDims.x) / float(data.defaultTexDims.y));
    mpTrace.reset();
    mpShadow.reset();
    mpHitDistance.reset();
    mReset = true;
}

void BakedGPISPass::setScene(RenderContext*, const ref<Scene>& scene)
{
    mpScene = scene;
    mReset = true;
}

void BakedGPISPass::execute(RenderContext* context, const RenderData& data)
{
    if (mReset)
    {
        auto& dict = data.getDictionary();
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[kRenderPassRefreshFlags] = flags | RenderPassRefreshFlags::RenderOptionsChanged;
        mFrameIndex = 0;
        mReset = false;
    }
    const auto color = data.getTexture("color");
    const auto transmission = data.getTexture("transmittance");
    if (!color && !transmission) return;
    if (!mpMean)
    {
        if (color) context->clearUAV(color->getUAV().get(), float4(.22f, .30f, .42f, 1.f));
        if (transmission) context->clearUAV(transmission->getUAV().get(), float4(1.f));
        return;
    }
    const auto& output = color ? color : transmission;
    const uint2 size(output->getWidth(), output->getHeight());
    if (!mpTrace)
    {
        constexpr char kShader[] = "RenderPasses/LatticeGPIS/BakedGPIS.cs.slang";
        DefineList defines;
        defines.add("WRITE_RADIANCE", color ? "1" : "0");
        defines.add("WRITE_TRANSMITTANCE", transmission ? "1" : "0");
        defines.add("SHADOW_STAGE", "0");
        mpTrace = ComputePass::create(mpDevice, kShader, "main", defines);
        if (color)
        {
            defines.add("SHADOW_STAGE", "1");
            mpShadow = ComputePass::create(mpDevice, kShader, "main", defines);
            mpHitDistance = mpDevice->createTexture2D(
                size.x, size.y, ResourceFormat::R32Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
            );
        }
    }
    const auto& camera = (mpScene ? mpScene->getCamera() : mpFallbackCamera)->getData();
    const uint32_t seed = mSeed + mFrameIndex++;
    auto bind = [&](const ShaderVar& vars)
    {
        mpMean->bind(vars["gDensity"]);
        vars["gFrameDim"] = size;
        vars["gSeed"] = seed;
        vars["gCameraPosition"] = camera.posW;
        vars["gCameraU"] = camera.cameraU;
        vars["gCameraV"] = camera.cameraV;
        vars["gCameraW"] = camera.cameraW;
        vars["gTranslation"] = mTranslation;
        vars["gWorldScale"] = mpMean->getBakedWorldScale();
        vars["gEll"] = mpMean->getBakedCorrelationLength();
    };
    auto vars = mpTrace->getRootVar();
    bind(vars);
    if (color) vars["gHitDistanceOutput"] = mpHitDistance;
    if (transmission) vars["gTransmission"] = transmission;
    mpTrace->execute(context, uint3(size, 1));
    if (color)
    {
        auto shadow = mpShadow->getRootVar();
        bind(shadow);
        shadow["gHitDistanceInput"] = mpHitDistance;
        shadow["gColor"] = color;
        mpShadow->execute(context, uint3(size, 1));
    }
}

void BakedGPISPass::renderUI(Gui::Widgets& widget)
{
    if (widget.button("Load baked GPIS..."))
    {
        std::filesystem::path file;
        if (openFileDialog({{"vdb", "Baked GPIS parameters"}}, file))
        {
            Properties props;
            props["cloudBakedFile"] = file;
            try { setProperties(props); }
            catch (const std::exception& e) { mError = e.what(); }
        }
    }
    if (mpMean)
    {
        widget.text(mFile.filename().string());
        widget.text(fmt::format("Baked correlation length {:.5g}, amplitude {:.5g}", mpMean->getBakedCorrelationLength(),
                                mpMean->getBakedAmplitude()));
    }
    if (!mError.empty()) widget.textWrapped(mError);
}
