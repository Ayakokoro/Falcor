# 方案

## 原始推导

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\int_A f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \tag{1}
$$

权重函数 $w(\mathbf{p}) = V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle$（即：沿视线可见、且具有光照余弦权重的表面测度）

$$
\hat{f}_v = \left[ \frac{\int_A f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_i) \cdot \overbrace{V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle}^{w(\mathbf{p})} \mathrm{d}\mathbf{p}}{\int_A \underbrace{V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle}_{w(\mathbf{p})} \mathrm{d}\mathbf{p}} \right] \cdot \left[ \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]
$$

**假设1：**材质分布 $f$ 与光照遮挡 $V_i$ 统计独立

得到

$$
\mathbb{E}_w[f \cdot V_i] \approx \mathbb{E}_w[f] \cdot \mathbb{E}_w[V_i]
$$

$$
\mathbb{E}_w[V_i] = \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_i) \cdot w(\mathbf{p})\mathrm{d}\mathbf{p}}{\int_A w(\mathbf{p})\mathrm{d}\mathbf{p}} = \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \equiv \hat{V}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \tag{5}
$$

$$
\mathbb{E}_w[f] \cdot \left[ \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right] = \frac{\int_A f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

**假设2：**在视线可见表面（$V_o = 1$）上的材质和法线统计特性与体素内所有表面（$A$）一致。

得到

$$
\approx \frac{\int_A f(\mathbf{p}) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \equiv \hat{f}_m(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \tag{4}
$$

将体素内的所有多边形表面 $A$ 按照法线所属的 4 个主导轴扇区（划分见下文）分割为 4 个子集 $A = \bigcup_{k=1}^4 A_k$： 

$$
\hat{f}_{\text{m}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \sum_{k=1}^4 \frac{\int_{A_k} f(\mathbf{p}) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

针对第 $k$ 个子集 $A_k$ 上的连续积分项：

$$
I_k = \frac{\int_{A_k} f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

对于**分子**，位置在区域 $A_k$ 上概率密度函数（PDF）为

$$
p(\mathbf{p}) = \frac{1}{\vert{}A_k\vert{}} = \frac{1}{\int_{A_k} 1 \, \mathrm{d}\mathbf{p}}, \quad \mathbf{p} \in A_k
$$

定义位置 $\mathbf{p} \sim p(\mathbf{p})$ 上的两个随机变量：

- **表面法线随机向量**：$\mathbf{N} = \mathbf{n}(\mathbf{p}) \in S^2$
- **材质参数随机向量**：$\boldsymbol{\Theta} = \boldsymbol{\theta}(\mathbf{p}) = (\rho(\mathbf{p}), \alpha(\mathbf{p}), \beta(\mathbf{p}), r(\mathbf{p}))$

数学期望分别为

$$
\mathbb{E}[\mathbf{N}] = \frac{1}{\vert{}A_k\vert{}} \int_{A_k} \mathbf{n}(\mathbf{p}) \mathrm{d}\mathbf{p} \implies \tilde{\mathbf{n}}_k = \frac{\mathbb{E}[\mathbf{N}]}{\Vert{}\mathbb{E}[\mathbf{N}]\Vert{}}
$$

$$
\mathbb{E}[\boldsymbol{\Theta}] = \frac{1}{\vert{}A_k\vert{}} \int_{A_k} \boldsymbol{\theta}(\mathbf{p}) \mathrm{d}\mathbf{p} \triangleq \bar{\boldsymbol{\theta}}_k
$$

对于给定的入射光方向 $\boldsymbol{\omega}_i$ 和出射方向 $\boldsymbol{\omega}_o$，定义关于法线 $\mathbf{n}$  和材质 $\boldsymbol{\theta}$ 的非线性多元函数 $g(\mathbf{n}, \boldsymbol{\theta})$：

$$
g(\mathbf{n}, \boldsymbol{\theta}) \triangleq f(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \mathbf{n}, \boldsymbol{\theta}) \cdot \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \cdot \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle
$$

$$
\int_{A_k} f(\mathbf{p},\boldsymbol{\omega_i},\boldsymbol{\omega_o}) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p} = \vert{}A_k\vert{} \int_{A_k} g(\mathbf{n}(\mathbf{p}), \boldsymbol{\theta}(\mathbf{p})) \frac{\mathrm{d}\mathbf{p}}{\vert{}A_k\vert{}} = \vert{}A_k\vert{} \cdot \mathbb{E}\Big[ g(\mathbf{N}, \boldsymbol{\Theta}) \Big]
$$

我们使用非线性函数的期望的一阶近似

$$
\mathbb{E}\Big[ g(\mathbf{N}, \boldsymbol{\Theta}) \Big] \approx g\Big( \mathbb{E}[\mathbf{N}], \mathbb{E}[\boldsymbol{\Theta}] \Big)
$$

- 误差来源：
    
    如果将复合函数 $g(\mathbf{X})$（其中 $\mathbf{X} = (\mathbf{N}, \boldsymbol{\Theta})$）在均值 $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ 处进行多元泰勒展开：
    
    $$
    g(\mathbf{X}) = g(\boldsymbol{\mu}) + \nabla g(\boldsymbol{\mu})^T (\mathbf{X} - \boldsymbol{\mu}) + \frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^T \mathbf{H}_g(\boldsymbol{\mu}) (\mathbf{X} - \boldsymbol{\mu}) + \mathcal{O}(\Vert{}\mathbf{X} - \boldsymbol{\mu}\Vert{}^3)
    $$
    
    两边取数学期望 $\mathbb{E}[\cdot]$：
    
    由于一阶导数项期望为 0（因为 $\mathbb{E}[\mathbf{X} - \boldsymbol{\mu}] = \mathbf{0}$），得到：
    
    $$
    \mathbb{E}[g(\mathbf{X})] = \underbrace{g(\mathbb{E}[\mathbf{X}])}_{\text{近似项}} + \underbrace{\frac{1}{2} \operatorname{Tr}\Big( \mathbf{H}_g(\boldsymbol{\mu}) \operatorname{Cov}(\mathbf{X}) \Big)}_{\text{误差主导项（二阶协方差修正）}} + \mathcal{O}(\dots)
    $$
    

即可得到

$$
f^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \tilde{\mathbf{n}}_k, \bar{\boldsymbol{\theta}}_k) \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_o \rangle \vert{}A_k\vert{} 
$$

也就是

$$
I_k \approx f^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \tilde{\mathbf{n}}_k, \bar{\boldsymbol{\theta}}_k) \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle \cdot \left[ \frac{\vert{}A_k\vert{} \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_o \rangle}{\int_A \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]
$$

$$
\frac{\vert{}A_k\vert{} \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_o \rangle}{\int_A \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \approx \frac{\vert{}A_k\vert{}}{\vert{}A\vert{}} \triangleq w_k \tag{2}
$$

此时即推导出了离散波瓣累加形式（单面、未做双面折叠）：

$$
\hat{f}_m(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \sum_{k=1}^4 w_k \left[ f_d^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) + f_s^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \right] \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle \tag{3}
$$

球面空间（上半球面，$y \ge 0$）根据直角坐标分量 $(x, z)$ 的**主导轴**被划分为 4 个扇区（对应 4 个波瓣）：  

- **X 主导（$\vert{}x\vert{} > \vert{}z\vert{}$，边界 $\vert{}x\vert{} \approx \vert{}z\vert{}$ 时按 $x \ge z$ 确定性归入 X）**：$x > 0$ → $k = 1$（$+X$），$x < 0$ → $k = 2$（$-X$）。
- **Z 主导（其余情况）**：$z > 0$ → $k = 3$（$+Z$），$z < 0$ → $k = 4$（$-Z$）。

假设多边形材质均为**双面材质**：每个多边形微元在法线向外 $\mathbf{n}$ 和向内 $-\mathbf{n}$ 上的光学属性与统计权重严格对称。因此聚合时先把法线折叠到上半球面（$-\mathbf{n}$ 归并到 $\mathbf{n}$），每个扇区波瓣实际上代表一对中心对称的法线 $\{\tilde{\mathbf{n}}_k, -\tilde{\mathbf{n}}_k\}$。将每个扇区波瓣按其对 $\pm\tilde{\mathbf{n}}_k$ 的对称性展开，使用仅存储上半球面的 4 个波瓣表达为：

$$
\hat{f}_m(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \sum_{k=1}^4 w_k \left( \left[ f_d^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \tilde{\mathbf{n}}_k) + f_s^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \tilde{\mathbf{n}}_k) \right] \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle + \left[ f_d^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; -\tilde{\mathbf{n}}_k) + f_s^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; -\tilde{\mathbf{n}}_k) \right] \langle -\tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle \right) \tag{3}
$$

式 (3) 可以进一步写为由有效法线 $\mathbf{n}_k^*(\boldsymbol{\omega}_i)$ 表示的等价形式：

$$
\hat{f}_m(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \sum_{k=1}^4 w_k \left[ f_d^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \mathbf{n}_k^*(\boldsymbol{\omega}_i)) + f_s^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \mathbf{n}_k^*(\boldsymbol{\omega}_i)) \right] \vert{}\tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i\vert{} \tag{4}
$$

其中

$$
\mathbf{n}_k^*(\boldsymbol{\omega}_i) = \operatorname{sgn}(\tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i) \, \tilde{\mathbf{n}}_k
$$

具体参数离散计算公式：

- **波瓣权重 $w_k$**：
    
    $$
    w_k = \frac{\vert{}A_k\vert{}}{\vert{}A\vert{}} = \frac{\sum_{j \in \text{Sector}_k} \vert{}P_j\vert{}}{\sum_{j=1}^M \vert{}P_j\vert{}} \tag{4}
    $$
    
- **波瓣平均法线 $\tilde{\mathbf{n}}_k$**（自身单位化归一）：
    
    $$
    \tilde{\mathbf{n}}_k = \frac{\sum_{j \in \text{Sector}_k} \vert{}P_j\vert{} \mathbf{n}_j}{\left\Vert{} \sum_{j \in \text{Sector}_k} \vert{}P_j\vert{} \mathbf{n}_j \right\Vert{}} \tag{5}
    $$
    
- **平均材质参数 $\bar{\boldsymbol{\theta}}_k = (\bar{\rho}_k, \bar{\alpha}_k, \bar{\beta}_k, \bar{r}_k)$**：
    
    $$
    \bar{\boldsymbol{\theta}}_k = \frac{\sum_{j \in \text{Sector}_k} \vert{}P_j\vert{} \cdot \text{MipmapQuery}(P_j, \text{Texture}_\theta)}{\vert{}A_k\vert{}} \tag{6}
    $$
    

## 包含对视线可见性的材质推导

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\int_A f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \tag{1}
$$

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \underbrace{\left[ \frac{\int_A f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]}_{\hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \quad \text{（目标可见材质项）}} \cdot \underbrace{\left[ \frac{\int_A f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]}_{R(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \quad \text{（待化简项）}} \tag{2}
$$

在体素表面  $A$  上，以视线可见投影面元定义归一化的概率密度函数（PDF）：

$$
p(\mathbf{p} \mid \boldsymbol{\omega}_o) = \frac{V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle}{\int_A V(\mathbf{q}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{q}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{q}} = \frac{\mathrm{d}A_o^{\perp,\text{vis}}(\mathbf{p})}{A_{\text{vis}}^\perp(\boldsymbol{\omega}_o)}
$$

**材质光学响应变量 $X$**：

$$
X(\mathbf{p}) = f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_i \rangle
$$

**入射光自遮挡指示变量 $Y$**：

$$
Y(\mathbf{p}) = V(\mathbf{p}, \boldsymbol{\omega}_i) \in \{0, 1\}
$$

原始 VABSDF 为联合期望：

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \int_A X(\mathbf{p}) Y(\mathbf{p}) \, p(\mathbf{p} \mid \boldsymbol{\omega}_o) \mathrm{d}\mathbf{p} = \mathbb{E}_{\boldsymbol{\omega}_o}[X \cdot Y]
$$

$$
\hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \int_A X(\mathbf{p}) \, p(\mathbf{p} \mid \boldsymbol{\omega}_o) \mathrm{d}\mathbf{p} = \mathbb{E}_{\boldsymbol{\omega}_o}[X]
$$

$$
R(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\mathbb{E}_{\boldsymbol{\omega}_o}[X \cdot Y]}{\mathbb{E}_{\boldsymbol{\omega}_o}[X]}
$$

假设：**在视线可见的所有微观表面上，材质光学响应 $X$（反照率、粗糙度、法线分布）与入射光方向的几何自遮挡 $Y$ 统计无关**：

$$
\operatorname{Cov}_{\boldsymbol{\omega}_o}(X, Y) = 0 \iff \mathbb{E}_{\boldsymbol{\omega}_o}[X \cdot Y] \approx \mathbb{E}_{\boldsymbol{\omega}_o}[X] \cdot \mathbb{E}_{\boldsymbol{\omega}_o}[Y]
$$

代入 $R(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$ 中：

$$
R(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \frac{\mathbb{E}_{\boldsymbol{\omega}_o}[X] \cdot \mathbb{E}_{\boldsymbol{\omega}_o}[Y]}{\mathbb{E}_{\boldsymbol{\omega}_o}[X]} = \mathbb{E}_{\boldsymbol{\omega}_o}[Y]
$$

将 $\mathbb{E}_{\boldsymbol{\omega}_o}[Y]$ 重新展开为积分形式，即得到无材质干扰的**视线受光率项 $\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$**：

$$
\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \triangleq \mathbb{E}_{\boldsymbol{\omega}_o}\left[ V(\mathbf{p}, \boldsymbol{\omega}_i) \right] = \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

从原始 VABSDF 中分离出 $\hat{f}_{\text{vis}}$ 的完整公式为：

$$
\hat{f}_v(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \cdot \hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \tag{3}
$$

$$
\hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\int_A f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \tag{4}
$$

$$
\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\int_A V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \tag{5}
$$

### **对于视线方向可见性材质项**

仍然根据面片法线拆分成多个 lobe。**注意：此处不做双面折叠**——可见性权重 $V(\mathbf{p}, \boldsymbol{\omega}_o)\langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle$ 对 $\mathbf{n} \to -\mathbf{n}$ 不对称（背向 $\boldsymbol{\omega}_o$ 的面片不参与可见投影），故须在全球面按法线 8 个卦限拆分，而非上半球 4 扇区。第 $k$ 个波瓣子集 $A_k$ 上的积分：

$$
I_k(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\int_{A_k} f(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A} V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

$$
I_k(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \underbrace{\left[ \frac{\int_{A_k} V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A} V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]}_{w_k(\boldsymbol{\omega}_o) \text{（波瓣可见权重）}} \cdot \underbrace{\left[ \frac{\int_{A_k} f(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_i \rangle \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A_k} V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \right]}_{\mathbf{f^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o)) \langle \tilde{\mathbf{n}}_k \cdot \boldsymbol{\omega}_i \rangle \text{（波瓣内视线可见材质）}}}
$$

对于单个波瓣项 $\hat{f}_{\text{vis}}^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$。在子集 $A_k$ 上，定义**视线可见条件概率密度测度**：

$$
\mathrm{d}\mu_k(\mathbf{p} \mid \boldsymbol{\omega}_o) = \frac{V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\text{VPA}_k(\boldsymbol{\omega}_o)}
$$

此时波瓣项即为该测度下的数学期望：

$$
\hat{f}_{\text{vis}}^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \mathbb{E}_{\mathbf{p} \sim \mu_k}\Big[ f\big(\mathbf{p}, \boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \mathbf{n}(\mathbf{p}), \boldsymbol{\theta}(\mathbf{p})\big) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_i \rangle \Big] \tag{6}
$$

用一阶泰勒展开近似

**代表性法线:**

$$
\tilde{\mathbf{n}}_k(\boldsymbol{\omega}_o) \triangleq \mathbb{E}_{\mathbf{p} \sim \mu_k}[\mathbf{n}(\mathbf{p})] = \frac{\int_{A_k} \mathbf{n}(\mathbf{p}) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A_k} \mathbf{V(\mathbf{p}, \boldsymbol{\omega}_o)} \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}\tag{7}
$$

**视线可见材质参数**：

$$
\bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o) \triangleq \mathbb{E}_{\mathbf{p} \sim \mu_k}[\boldsymbol{\theta}(\mathbf{p})] = \frac{\int_{A_k} \boldsymbol{\theta}(\mathbf{p}) \mathbf{V(\mathbf{p}, \boldsymbol{\omega}_o)} \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A_k} \mathbf{V(\mathbf{p}, \boldsymbol{\omega}_o)} \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}} \tag{8}
$$

$$
\hat{f}_{\text{vis}}^{(k)}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx f_{\text{micro}}\left(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \tilde{\mathbf{n}}_k(\boldsymbol{\omega}_o), \bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o)\right) \langle \tilde{\mathbf{n}}_k(\boldsymbol{\omega}_o) \cdot \boldsymbol{\omega}_i \rangle \tag{6}
$$

最终的近似式：

$$
\hat{f}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \sum_{k=1}^8 w_k(\boldsymbol{\omega}_o) \cdot \left[ f_d^{(k)}\left(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o)\right) + f_s^{(k)}\left(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o; \bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o)\right) \right] \langle \tilde{\mathbf{n}}_k(\boldsymbol{\omega}_o) \cdot \boldsymbol{\omega}_i \rangle \tag{7}
$$

但是这样的材质和法线仍然是方向有关的，我们可以进一步假设当视线 $\boldsymbol{\omega}_o$ 在该卦限内扫过时，体素内部微观几何的**相对遮挡关系主要由该波瓣的主视角 $\mathbf{n}_{0, k}$ 决定：**

$$
\mathbf{n}_{0, k} \triangleq \frac{\int_{A_k} \mathbf{n}(\mathbf{p}) \mathrm{d}\mathbf{p}}{\left\Vert{} \int_{A_k} \mathbf{n}(\mathbf{p}) \mathrm{d}\mathbf{p} \right\Vert{}}
$$

$$
V(\mathbf{p}, \boldsymbol{\omega}_o) \approx V(\mathbf{p}, \mathbf{n}_{0, k}) \equiv V_k^*(\mathbf{p}) \in \{0, 1\} \quad (\forall \boldsymbol{\omega}_o \in \Omega_k)
$$

基于该假设，体素内的几何表面在第 $k$ 卦限内被严格划分为**可见几何子集**与**隐藏几何子集**：

$$
A_k^{\text{vis}} \triangleq \{ \mathbf{p} \in A_k \mid V_k^*(\mathbf{p}) = 1 \}
$$

将 $V(\mathbf{p}, \boldsymbol{\omega}_o) \approx V_k^*(\mathbf{p})$ 代入条件期望材质公式中：

$$
\bar{\boldsymbol{\theta}}_k(\boldsymbol{\omega}_o) = \frac{\int_{A_k} \boldsymbol{\theta}(\mathbf{p}) V_k^*(\mathbf{p}) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}{\int_{A_k} V_k^*(\mathbf{p}) \langle \mathbf{n}(\mathbf{p}) \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p}}
$$

### 对于可见性项

分母是沿视线 $\boldsymbol{\omega}_o$ 可见的投影面积 $\text{VPA}(\boldsymbol{\omega}_o)$。在微表面理论中，它等于**总投影面积 $\text{TPA}(\boldsymbol{\omega}_o)$ 乘以单向无遮蔽概率 $G_1(\boldsymbol{\omega}_o)$**：

$$
\int_A V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p} = \text{TPA}(\boldsymbol{\omega}_o) \cdot G_1(\boldsymbol{\omega}_o)
$$

分子是同时被视线 $\boldsymbol{\omega}_o$ 看到且被入射光 $\boldsymbol{\omega}_i$ 照亮的微表面沿 $\boldsymbol{\omega}_o$ 的投影面积。在微表面理论中，它等于**总投影面积 $\text{TPA}(\boldsymbol{\omega}_o)$ 乘以双向联合无遮挡概率 $G_2(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)$**：

$$
\int_A V(\mathbf{p}, \boldsymbol{\omega}_i) V(\mathbf{p}, \boldsymbol{\omega}_o) \langle \mathbf{n} \cdot \boldsymbol{\omega}_o \rangle \mathrm{d}\mathbf{p} \approx \text{TPA}(\boldsymbol{\omega}_o) \cdot G_2(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)
$$

则

$$
\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{\text{TPA}(\boldsymbol{\omega}_o) \cdot G_2(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)}{\text{TPA}(\boldsymbol{\omega}_o) \cdot G_1(\boldsymbol{\omega}_o)} = \frac{G_2(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o)}{G_1(\boldsymbol{\omega}_o)}
$$

在 **Heitz (2014) 高度相关 Smith 假设**下：

$$
G_1(\boldsymbol{\omega}) = \frac{1}{1 + \Lambda(\boldsymbol{\omega})} \iff \Lambda(\boldsymbol{\omega}) = \frac{\text{TPA}(\boldsymbol{\omega}) - \text{VPA}(\boldsymbol{\omega})}{\text{VPA}(\boldsymbol{\omega})}
$$

$$
G_2(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) = \frac{1}{1 + \Lambda(\boldsymbol{\omega}_i) + \Lambda(\boldsymbol{\omega}_o)}
$$

两项相除，直接得到该视线受光率项的 **Smith 解析近似式**：

$$
\hat{V}_{\text{vis}}(\boldsymbol{\omega}_i, \boldsymbol{\omega}_o) \approx \frac{1 + \Lambda(\boldsymbol{\omega}_o)}{1 + \Lambda(\boldsymbol{\omega}_i) + \Lambda(\boldsymbol{\omega}_o)}
$$

- **视线完全无阻挡（$\Lambda(\boldsymbol{\omega}_o) = 0$）**：
    
    $$
    \hat{V}_{\text{vis}} = \frac{1 + 0}{1 + \Lambda(\boldsymbol{\omega}_i) + 0} = \frac{1}{1 + \Lambda(\boldsymbol{\omega}_i)} = \frac{\text{VPA}(\boldsymbol{\omega}_i)}{\text{TPA}(\boldsymbol{\omega}_i)}
    $$
    
    完全退化为原论文中的单向投影受光率公式。
    
- **入射光完全无阻挡（$\Lambda(\boldsymbol{\omega}_i) = 0$）**：
    
    $$
    \hat{V}_{\text{vis}} = \frac{1 + \Lambda(\boldsymbol{\omega}_o)}{1 + 0 + \Lambda(\boldsymbol{\omega}_o)} \equiv 1.0
    $$
    
- **双层紧贴几何（如海报正视）**：
    
    视线与光线方向均暴露最外层，$\Lambda_i \approx 0, \Lambda_o \approx 0 \implies \hat{V}_{\text{vis}} \approx 1.0$，消除了原论文中因分母包含背部墙体而产生的 50% 能量暗斑。