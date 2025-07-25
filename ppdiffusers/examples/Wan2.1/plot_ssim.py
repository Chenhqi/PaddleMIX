import os
import matplotlib
import matplotlib.pyplot as plt

# ---------- 鏁版嵁 ----------
records = [
    # ("tgate",                     1.20, 0.740),
    # ("pab",                       1.57, 0.8516),
    # ("blockdance",                     2.10, 0.858),
    # ("teacache",                  1.73, 0.797),
    # ("firstblock_taylorseer 0.07", 2.03, 0.917),
    # ("firstblock_taylorseer 0.14", 3.26, 0.799),
    # ("teablockcache 0.0", 0.93, 1.0),
    # ("teablockcache 0.3", 1.34, 0.9833),
    # ("teablockcache 0.9", 1.66, 0.9385),
    # ("teablockcache_taylor 2", 3.54, 0.7663),
    # ("teablockcache_taylor 1", 2.72, 0.8665),
    # ("teablockcache_taylor 0.5", 2.00, 0.9364),
    # ("sortblockcache", 1.66, 0.95176),
    # ("sortblockcache-fit-taylor", 1.81, 0.9520),
    # ("sortblockcache-fit-taylor", 2.00, 0.9520),
    # ("sortblockcache-fit-taylor", 2.307, 0.91),
    # ("sortblockcache-fit-taylor", 2.88, 0.84326),
    # ("sortblockcache-fit-taylor", 3.06, 0.83731),
    # ("tgate",                     1.20, 0.740),
    # ("pab",                       1.57, 0.8516),
    # ("blockdance",                     2.10, 0.858),
    # ("teacache",                  1.73, 0.797),
    # ("firstblock_taylorseer 0.07", 2.03, 0.917),
    # ("firstblock_taylorseer 0.14", 3.26, 0.799),
    # ("teablockcache 0.0", 0.93, 1.0),
    # ("teablockcache 0.3", 1.34, 0.9833),
    # ("teablockcache 0.9", 1.66, 0.9385),
    # ("teablockcache_taylor 2", 3.54, 0.7663),
    # ("teablockcache_taylor 1", 2.72, 0.8665),
    # ("teablockcache_taylor 0.5", 2.00, 0.9364),
    # ("sortblockcache", 1.66, 0.95176),
    # ("sortblockcache-fit-taylor", 1.81, 0.9520),
    # ("sortblockcache-fit-taylor", 2.00, 0.9520),
    # ("sortblockcache-fit-taylor", 2.307, 0.91),
    # # ("sortblockcache-fit-taylor", 2.88, 0.84326),
    # ("sortblockcache-fit-taylor", 3.06, 0.83731),
    ("tgate",                     1.20, 0.740),
    ("tgate",                     1.60, 0.7),
    ("tgate",                     1.82, 0.559),
    ("pab",                       1.37, 0.8516),
    ("pab",                       1.57, 0.849),
    ("pab",                       1.83, 0.485),
    ("taylorseer",                     1.71, 0.850),
    ("taylorseer",                     2.04, 0.763),
    ("taylorseer",                     2.83, 0.648),
    ("teacache",                  1.44, 0.8114),
    ("teacache",                  1.73, 0.805),
    ("teacache",                  2.24, 0.717),
    ("sortblock", 1.66, 0.97727),
    ("sortblock", 1.79, 0.96618),
    ("sortblock", 2.00, 0.9520),
    ("sortblock", 2.21, 0.91984),
    # ("sortblockcache-fit-taylor", 2.88, 0.84326),
    ("sortblock", 3.06, 0.83731),
]
# --------------------------

# 鏃犲浘褰㈢晫闈㈡椂鍒囨崲鍒伴潪浜や簰鍚庣
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 鎸夋ā鍨嬪垎缁�
groups = {}
for full_name, spd, ssim in records:
    base = full_name.split()[0]          # 鍙栫┖鏍煎墠閮ㄥ垎浣滀负鈥滄ā鍨嬪悕鈥�
    groups.setdefault(base, []).append((spd, ssim, full_name))

fig, ax = plt.subplots(figsize=(7, 4))

for base, pts in groups.items():
    pts.sort(key=lambda x: x[0])         # 鎸夊姞閫熸瘮鍗囧簭锛屼究浜庣敾绾�
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    scatter = ax.scatter(xs, ys, label=base)     # 鏁ｇ偣
    if len(pts) > 1:
        ax.plot(xs, ys, color=scatter.get_facecolors()[0])  # 杩炵嚎

    # for x, y, lbl in pts:               # 缁欐瘡鐐瑰姞瀹屾暣鏍囩
    #     ax.text(x, y, lbl, fontsize=8, ha="right", va="bottom")

ax.set_xlabel("Speed-up")
ax.set_ylabel("SSIM")
ax.set_title("Speed-up vs. Image Quality (SSIM)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Model", fontsize=7)
plt.tight_layout()

# 鏈� / 鏃� GUI 鐨勪袱绉嶈緭鍑烘柟寮�
if matplotlib.get_backend() == "Agg":
    out_file = "accel_vs_ssim.png"
    plt.savefig(out_file, dpi=150)
    print(f"鏃犲浘褰㈢晫闈細鍥惧凡淇濆瓨涓� {out_file}")
else:
    plt.show()