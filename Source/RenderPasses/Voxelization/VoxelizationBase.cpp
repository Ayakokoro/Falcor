#include "VoxelizationBase.h"
#include "VoxelizationPass.h"
#include "RayMarchingPass.h"
#include "Core/Platform/OS.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RayMarchingPass>();
    registry.registerClass<RenderPass, VoxelizationPass>();
}

uint3 VoxelizationBase::MinFactor = uint3(1, 1, 1);
bool VoxelizationBase::FileUpdated = true;
bool VoxelizationBase::LightChanged = true;
// Keep the trailing separator for the voxel writer's filename concatenation.
std::string VoxelizationBase::ResourceFolder = (getProjectDirectory() / "resource").generic_string() + "/";
GridData VoxelizationBase::GlobalGridData = {};

std::random_device rd;
std::mt19937 Random::Generator{ rd() };
std::uniform_real_distribution<double> Random::Distribution{ 0.0, 1.0 };

const std::string kInputDiffuse = "diffuse";
const std::string kInputSpecular = "specular";
const std::string kInputRoughness = "roughness";
const std::string kInputArea = "area";
