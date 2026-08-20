#define S_MAX 23
#define EPSILON exp2(-float(S_MAX))

// 参数化无排序八叉树遍历
HitResult rayMarchingParametric(float3 from, float3 direction, float maxDistance, RayCone cone)
{
    // --- 1. 预处理光线方向，避免除零异常 ---
    direction = normalizeSafe(direction);
    float3 d = direction;
    if (abs(d.x) < EPSILON) d.x = sign(d.x) * EPSILON;
    if (abs(d.y) < EPSILON) d.y = sign(d.y) * EPSILON;
    if (abs(d.z) < EPSILON) d.z = sign(d.z) * EPSILON;

    // --- 2. 坐标系镜像化 (强制转换为纯负向遍历) ---
    uint rootSize = 1u << octreeMaxDepth;
    // 映射到 [1, 2] 规范化空间
    float3 normOrigin = 1.0f + (from / float(rootSize));
    normOrigin = clamp(normOrigin, 1.0f, 2.0f - EPSILON);

    // ray_coef 取负值，因为我们强制光线向负方向行进
    float3 ray_coef = -1.0f / abs(d);
    float3 normOriginMirrored = normOrigin;

    // octant_mask 记录了哪些轴原本是正向的，被我们强制镜像了
    uint octant_mask = 0;

    // 如果原始方向是正向的，则镜像它 (3.0 - x)，使其在计算中表现为负向
    if (d.x > 0.0f) { octant_mask |= 1; normOriginMirrored.x = 3.0f - normOrigin.x; }
    if (d.y > 0.0f) { octant_mask |= 2; normOriginMirrored.y = 3.0f - normOrigin.y; }
    if (d.z > 0.0f) { octant_mask |= 4; normOriginMirrored.z = 3.0f - normOrigin.z; }

    // 预计算射线偏置 (Ray Bias)
    float3 ray_bias = -normOriginMirrored * ray_coef;

    // --- 3. 根节点求交 (注意：进入点为 2.0，退出点为 1.0) ---
    // 因为 d 为负向，所以先触碰 2.0 平面，后触碰 1.0 平面
    float3 t_entry = 2.0f * ray_coef + ray_bias;
    float3 t_exit  = 1.0f * ray_coef + ray_bias;

    float t_min = max(max(t_entry.x, t_entry.y), t_entry.z);
    float t_max = min(min(t_exit.x,  t_exit.y),  t_exit.z);

    t_min = max(t_min, 0.0f);
    if (maxDistance > 0) t_max = min(t_max, maxDistance / float(rootSize));
    if (t_min > t_max) return { false, int3(0), float3(0), 0, 0 };

    // --- 4. 初始化状态机 ---
    int2 stack[S_MAX + 1];
    float h = t_max;

    uint parent = 0;
    uint idx = 0;
    float3 pos = float3(1.0f); // 当前层级体素在镜像空间下的 min 角
    int scale = S_MAX - 1;
    float scale_exp2 = 0.5f;

    // 判断进入根节点时，优先落入哪个子空间
    // t_center 是根节点中心点 (1.5) 的 t 值。如果进入整个网格 (t_min) 时还没穿过中心点，
    // 说明此时坐标必然 > 1.5，因此落在上层半区。
    float3 t_center = 1.5f * ray_coef + ray_bias;
    if (t_center.x > t_min) { idx ^= 1; pos.x = 1.5f; }
    if (t_center.y > t_min) { idx ^= 2; pos.y = 1.5f; }
    if (t_center.z > t_min) { idx ^= 4; pos.z = 1.5f; }

    int stop_scale = S_MAX - octreeMaxDepth;

    // --- 5. 核心状态机循环 ---
    while (scale >= stop_scale)
    {
        // 计算光线离开当前体素边界的 t 值
        // 由于是负向遍历，退出当前体素必定是穿过该体素的最小角 (pos)
        float3 t_corner = pos * ray_coef + ray_bias;
        float tc_max = min(min(t_corner.x, t_corner.y), t_corner.z);

        OctreeNode node = octreeBuffer[parent];
        uint child_shift = idx ^ octant_mask; // 还原真实内存中的 idx
        bool childExists = (node.childMask & (1u << child_shift)) != 0;

        // 【INTERSECT】
        if (childExists && t_min <= t_max)
        {
            uint nodeSize = 1u << (scale - stop_scale);
            float footprint = cone.bias + tc_max * cone.tanHalfAngle;
            uint currentDepth = S_MAX - scale;

            bool forceStop = (forcedLOD > 0 && currentDepth == forcedLOD);
            bool lodStop = ((maxLODLevel < 0 || currentDepth <= maxLODLevel) && footprint > (float)nodeSize);

            if (scale == stop_scale || forceStop || lodStop)
            {
                // -- 命中 --
                uint gbIndex = node.dataIndex;
                float exactHitT = t_min * float(rootSize);
                float3 exactHitWorld = from + exactHitT * direction;

                // 坐标解压：还原回 LOD-0 的整数格位置
                float3 hitPosOrig = pos;
                // 如果原本光线在该轴是正向的，我们在前边镜像了它，现在还原回来
                // 还原公式：原始的 min 角 = 3.0 - (镜像空间下的 min 角 + 体素尺寸)
                if ((octant_mask & 1) != 0) hitPosOrig.x = 3.0f - scale_exp2 - pos.x;
                if ((octant_mask & 2) != 0) hitPosOrig.y = 3.0f - scale_exp2 - pos.y;
                if ((octant_mask & 4) != 0) hitPosOrig.z = 3.0f - scale_exp2 - pos.z;

                float3 hitCell = (hitPosOrig - 1.0f) * float(rootSize);
                int3 cellInt = int3(floor(hitCell + RAY_BIAS));

                if (!outsideGrid(cellInt)) {
                    // Ellipsoid 裁剪逻辑可以放这里
                    return { true, cellInt, exactHitWorld, gbIndex, nodeSize };
                }
            }

            // 【PUSH】
            float tv_max = min(t_max, tc_max);
            float half_scale = scale_exp2 * 0.5f;

            // 计算当前体素中心平面的 t 值
            float3 t_mid = (pos + half_scale) * ray_coef + ray_bias;

            if (tc_max < h)
            {
                stack[scale] = int2(parent, asint(t_max));
                h = tc_max;
            }

            parent = node.childBase + countbits(node.childMask & ((1u << child_shift) - 1));

            idx = 0;
            scale--;
            scale_exp2 = half_scale;

            // 由于向负向遍历，如果中心平面的交点 t_mid 大于光线当前的 t_min，
            // 意味着光线还没有穿过中心平面，因此必须落在大于中心坐标的上半区。
            if (t_mid.x > t_min) { idx ^= 1; pos.x += scale_exp2; }
            if (t_mid.y > t_min) { idx ^= 2; pos.y += scale_exp2; }
            if (t_mid.z > t_min) { idx ^= 4; pos.z += scale_exp2; }

            t_max = tv_max;
            continue;
        }

        // 【ADVANCE】
        uint step_mask = 0;
        // 光线通过 min 面离开当前体素，意味着下一个兄弟体素的坐标必定更小
        if (t_corner.x <= tc_max) { step_mask |= 1; pos.x -= scale_exp2; }
        if (t_corner.y <= tc_max) { step_mask |= 2; pos.y -= scale_exp2; }
        if (t_corner.z <= tc_max) { step_mask |= 4; pos.z -= scale_exp2; }

        t_min = tc_max;
        idx ^= step_mask;

        // 【POP】
        if ((idx & step_mask) != 0)
        {
            uint differing_bits = 0;
            // pos 已经被减去了 scale_exp2，故通过异或旧坐标 (pos + scale_exp2) 来找差异位
            if ((step_mask & 1) != 0) differing_bits |= asuint(pos.x) ^ asuint(pos.x + scale_exp2);
            if ((step_mask & 2) != 0) differing_bits |= asuint(pos.y) ^ asuint(pos.y + scale_exp2);
            if ((step_mask & 4) != 0) differing_bits |= asuint(pos.z) ^ asuint(pos.z + scale_exp2);

            scale = (firstbithigh(differing_bits) - 23) + 127;
            if (scale >= S_MAX) return { false, int3(0), float3(0), 0, 0 }; // 光线溢出根节点

            scale_exp2 = asfloat((scale - 127) << 23);

            int2 stackEntry = stack[scale];
            parent = (uint)stackEntry.x;
            t_max = asfloat(stackEntry.y);

            uint shx = asuint(pos.x) >> scale;
            uint shy = asuint(pos.y) >> scale;
            uint shz = asuint(pos.z) >> scale;
            pos = float3(asfloat(shx << scale), asfloat(shy << scale), asfloat(shz << scale));
            idx = (shx & 1) | ((shy & 1) << 1) | ((shz & 1) << 2);

            h = 0.0f;
        }
    }

    return { false, int3(0), float3(0), 0, 0 };
}
