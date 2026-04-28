import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

# ============================================================
# Config
# ============================================================

OUT_DIR = Path(r"D:\Research\Result\Others\SH")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 球面采样分辨率（用于积分和球面函数评估）
N_THETA = 721
N_PHI = 1441

# 半球圆盘图分辨率
DISK_RES = 601

# ============================================================
# SH basis (even orders only: l = 0, 2, 4)
# 与你的 calcEvenSH 保持一致
# ============================================================

SQRTPI = np.sqrt(np.pi)


def calc_even_sh(idx, x, y, z):
    x2 = x * x
    y2 = y * y
    z2 = z * z

    # l = 0
    if idx == 0:
        return np.ones_like(x) * (1.0 / (2.0 * SQRTPI))  # m = 0

    # l = 2
    elif idx == 1:
        return x * y * np.sqrt(15.0) / (2.0 * SQRTPI)  # m = -2
    elif idx == 2:
        return y * z * np.sqrt(15.0) / (2.0 * SQRTPI)  # m = -1
    elif idx == 3:
        return (3.0 * z2 - 1.0) * np.sqrt(5.0) / (4.0 * SQRTPI)  # m = 0
    elif idx == 4:
        return x * z * np.sqrt(15.0) / (2.0 * SQRTPI)  # m = 1
    elif idx == 5:
        return (x2 - y2) * np.sqrt(15.0) / (4.0 * SQRTPI)  # m = 2

    # l = 4
    elif idx == 6:
        return x * y * (x2 - y2) * 3.0 * np.sqrt(35.0) / (4.0 * SQRTPI)  # m = -4
    elif idx == 7:
        return y * z * (3.0 * x2 - y2) * 3.0 * np.sqrt(70.0) / (8.0 * SQRTPI)  # m = -3
    elif idx == 8:
        return x * y * (7.0 * z2 - 1.0) * 3.0 * np.sqrt(5.0) / (4.0 * SQRTPI)  # m = -2
    elif idx == 9:
        return y * z * (7.0 * z2 - 3.0) * 3.0 * np.sqrt(10.0) / (8.0 * SQRTPI)  # m = -1
    elif idx == 10:
        return (35.0 * z2 * z2 - 30.0 * z2 + 3.0) * 3.0 / (16.0 * SQRTPI)  # m = 0
    elif idx == 11:
        return x * z * (7.0 * z2 - 3.0) * 3.0 * np.sqrt(10.0) / (8.0 * SQRTPI)  # m = 1
    elif idx == 12:
        return (x2 - y2) * (7.0 * z2 - 1.0) * 3.0 * np.sqrt(5.0) / (8.0 * SQRTPI)  # m = 2
    elif idx == 13:
        return x * z * (x2 - 3.0 * y2) * 3.0 * np.sqrt(70.0) / (8.0 * SQRTPI)  # m = 3
    elif idx == 14:
        return (
            x2 * (x2 - 3.0 * y2) - y2 * (3.0 * x2 - y2)
        ) * 3.0 * np.sqrt(35.0) / (16.0 * SQRTPI)  # m = 4

    raise ValueError(f"Invalid SH index: {idx}")


BASIS_SETS = {
    "l=0": [0],
    "l=0,2": list(range(6)),
    "l=0,2,4": list(range(15)),
}

# ============================================================
# Ground truth
# 双面单位正方形，法线 n=(0,0,1)，面积 A=1
# f(w)=|n·w|=|z|
# ============================================================


def ground_truth(x, y, z):
    return np.abs(z)


# ============================================================
# Sphere sampling for integration
# ============================================================


def build_sphere_grid(n_theta=N_THETA, n_phi=N_PHI):
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    x = np.sin(Theta) * np.cos(Phi)
    y = np.sin(Theta) * np.sin(Phi)
    z = np.cos(Theta)

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    weights = np.sin(Theta) * dtheta * dphi

    return theta, phi, x, y, z, weights


# ============================================================
# SH projection / reconstruction
# ============================================================


def compute_even_sh_coefficients(f, x, y, z, weights, basis_count=15):
    basis_values = []
    coeffs = []

    for i in range(basis_count):
        Yi = calc_even_sh(i, x, y, z)
        ci = np.sum(f * Yi * weights)
        basis_values.append(Yi)
        coeffs.append(ci)

    return np.array(coeffs, dtype=np.float64), basis_values


def reconstruct(coeffs, basis_values, active_indices):
    f_hat = np.zeros_like(basis_values[active_indices[0]], dtype=np.float64)
    for i in active_indices:
        f_hat = f_hat + coeffs[i] * basis_values[i]
    return f_hat


# ============================================================
# Error metrics
# ============================================================


def relative_l2_error(f, f_hat, weights):
    num = np.sum(((f_hat - f) ** 2) * weights)
    den = np.sum((f ** 2) * weights)
    return np.sqrt(num / den)


def mae(f, f_hat, weights):
    return np.sum(np.abs(f_hat - f) * weights) / (4.0 * np.pi)


def max_abs_error(f, f_hat):
    return np.max(np.abs(f_hat - f))


# ============================================================
# Hemisphere disk visualization
# 只显示 z>=0 半球，圆心对应法线方向 (0,0,1)
# ============================================================


def make_hemisphere_disk_function(func, res=DISK_RES):
    lin = np.linspace(-1.0, 1.0, res)
    u, v = np.meshgrid(lin, lin, indexing="xy")
    r = np.sqrt(u * u + v * v)

    img = np.full((res, res), np.nan, dtype=np.float64)

    mask = r <= 1.0
    rr = r[mask]
    phi = np.arctan2(v[mask], u[mask])

    theta = rr * (0.5 * np.pi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    img[mask] = func(x, y, z)
    return img


# ============================================================
# Save single disk image (no text, no axes, no border, no circle)
# ============================================================


def save_single_disk_map(
    image,
    out_path,
    cmap="viridis",
    vmin=None,
    vmax=None,
    fig_height=4.0,
    fig_width=4.0,
):
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax.imshow(
        image,
        origin="lower",
        extent=[-1, 1, -1, 1],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()

    fig.savefig(
        out_path,
        dpi=220,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(fig)


def save_disk_map_series(
    images,
    out_paths,
    cmap="viridis",
):
    assert len(images) == len(out_paths)

    vmin = min(np.nanmin(img) for img in images)
    vmax = max(np.nanmax(img) for img in images)

    for img, path in zip(images, out_paths):
        save_single_disk_map(
            img,
            path,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    return vmin, vmax


# ============================================================
# Save standalone colorbar only
# ============================================================


def save_standalone_colorbar(
    out_path,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    fig_height=4.0,
    fig_width=0.5,
):
    fig = plt.figure(figsize=(fig_width, fig_height), frameon=False)
    cax = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax)

    # 去掉所有文字、数字、刻度
    cbar.set_ticks([])
    cbar.set_label("")
    cbar.ax.tick_params(
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
        bottom=False,
        top=False,
        labelbottom=False,
        length=0,
    )

    # 去掉边框
    cbar.outline.set_visible(False)
    cbar.ax.set_frame_on(False)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    fig.savefig(
        out_path,
        dpi=220,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def main():
    # 1) sphere integration grid
    theta, phi, x, y, z, weights = build_sphere_grid()

    # 2) ground truth
    f = ground_truth(x, y, z)

    # 3) SH coefficients
    coeffs, basis_values = compute_even_sh_coefficients(f, x, y, z, weights, basis_count=15)

    # 4) reconstructions
    recons = {}
    metrics = {}

    for name, active_indices in BASIS_SETS.items():
        f_hat = reconstruct(coeffs, basis_values, active_indices)
        recons[name] = f_hat
        metrics[name] = {
            "basis_count": len(active_indices),
            "relative_l2": relative_l2_error(f, f_hat, weights),
            "mae": mae(f, f_hat, weights),
            "max_abs": max_abs_error(f, f_hat),
        }

    # 5) print results
    print("Ground truth: f(ω) = |z|")
    print("Error metrics:")
    for name in ["l=0", "l=0,2", "l=0,2,4"]:
        m = metrics[name]
        print(
            f"{name:7s} | basis={m['basis_count']:2d} | "
            f"rel_L2={m['relative_l2']:.10f} | "
            f"MAE={m['mae']:.10f} | "
            f"MaxAbs={m['max_abs']:.10f}"
        )

    # 6) save text report
    report_path = OUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write("Ground truth: f(ω)=|z|\n")
        fout.write("Two-sided unit square with normal n=(0,0,1)\n\n")
        fout.write("Error metrics:\n")
        for name in ["l=0", "l=0,2", "l=0,2,4"]:
            m = metrics[name]
            fout.write(
                f"{name:7s} | basis={m['basis_count']:2d} | "
                f"rel_L2={m['relative_l2']:.10f} | "
                f"MAE={m['mae']:.10f} | "
                f"MaxAbs={m['max_abs']:.10f}\n"
            )

        fout.write("\nCoefficients:\n")
        for i, c in enumerate(coeffs):
            fout.write(f"c[{i}] = {c:.12f}\n")

    # 7) prepare disk maps
    truth_disk = make_hemisphere_disk_function(
        lambda xx, yy, zz: ground_truth(xx, yy, zz)
    )

    l0_disk = make_hemisphere_disk_function(
        lambda xx, yy, zz: reconstruct(
            coeffs,
            [calc_even_sh(i, xx, yy, zz) for i in range(15)],
            BASIS_SETS["l=0"],
        )
    )

    l02_disk = make_hemisphere_disk_function(
        lambda xx, yy, zz: reconstruct(
            coeffs,
            [calc_even_sh(i, xx, yy, zz) for i in range(15)],
            BASIS_SETS["l=0,2"],
        )
    )

    l024_disk = make_hemisphere_disk_function(
        lambda xx, yy, zz: reconstruct(
            coeffs,
            [calc_even_sh(i, xx, yy, zz) for i in range(15)],
            BASIS_SETS["l=0,2,4"],
        )
    )

    recon_images = [truth_disk, l0_disk, l02_disk, l024_disk]

    err_l0 = np.abs(l0_disk - truth_disk)
    err_l02 = np.abs(l02_disk - truth_disk)
    err_l024 = np.abs(l024_disk - truth_disk)

    err_images = [err_l0, err_l02, err_l024]

    # 8) save reconstruction images
    recon_paths = [
        OUT_DIR / "GT.png",
        OUT_DIR / "0.png",
        OUT_DIR / "02.png",
        OUT_DIR / "024.png",
    ]

    recon_vmin, recon_vmax = save_disk_map_series(
        recon_images,
        recon_paths,
        cmap="viridis",
    )

    # 9) save error images
    err_paths = [
        OUT_DIR / "0_Error.png",
        OUT_DIR / "02_Error.png",
        OUT_DIR / "024_Error.png",
    ]

    err_vmin, err_vmax = save_disk_map_series(
        err_images,
        err_paths,
        cmap="magma",
    )

    # 10) save standalone colorbars
    save_standalone_colorbar(
        OUT_DIR / "Colorbar.png",
        cmap="viridis",
        vmin=recon_vmin,
        vmax=recon_vmax,
        fig_height=4.0,
        fig_width=0.45,
    )

    save_standalone_colorbar(
        OUT_DIR / "Error_Colorbar.png",
        cmap="magma",
        vmin=err_vmin,
        vmax=err_vmax,
        fig_height=4.0,
        fig_width=0.45,
    )

    print(f"\nOutputs saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()