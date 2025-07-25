import os
import matplotlib
import matplotlib.pyplot as plt

# ---------- 鏁版嵁 ----------
# 璇存槑锛氱涓夊垪璇峰～鍏� *瀹為檯* coco1k-PSNR锛堟暟鍊艰秺楂樿秺濂斤級
records = [
    # ("tgate",                     1.20, 20.11),
    # ("pab",                       1.57, 24.67),
    # ("blockdance",                2.10, 24.65),
    # ("teacache",                  1.73, 21.68),
    # ("firstblock_taylorseer 0.07", 2.03, 27.92),
    # ("firstblock_taylorseer 0.14", 3.26, 21.74),
    # # ("teablockcache 0.0",    0.93, 100.00),
    # ("teablockcache 0.3",    1.34,  37.17),
    # ("teablockcache 0.9",    1.66,  29.81),
    # ("teablockcache_taylor 2",    3.99,  20.46),
    # ("teablockcache_taylor 1",    2.98,  24.47),
    # ("teablockcache_taylor 0.5",  2.13,  29.49),
    # ("sortblockcache",            1.66,  31.1875),
    # ("tgate",                     1.20, 19.96),
    # ("pab",                       1.57, 24.65),
    # ("blockdance",                2.10, 24.57),
    # ("teacache",                  1.73, 22.11),
    # ("firstblock_taylorseer 0.07", 2.03, 27.59),
    # ("firstblock_taylorseer 0.14", 3.26, 21.42),
    # # ("teablockcache 0.0",    0.93, 100.00),
    # ("teablockcache 0.3",    1.34,  37.17),
    # ("teablockcache 0.9",    1.66,  29.81),
    # ("teablockcache_taylor 2",    3.99,  20.46),
    # ("teablockcache_taylor 1",    2.98,  24.47),
    # ("teablockcache_taylor 0.5",  2.13,  29.49),
    # ("sortblockcache",            1.66,  31.1875),

    ("tgate",                     1.20, 19.96),
    ("tgate",                     1.60, 19.18),
    ("tgate",                     1.82, 16.37),
    ("pab",                       1.37, 27.60),
    ("pab",                       1.57, 24.65),
    ("pab",                       1.83, 13.4),
    ("taylorseer",                     1.71, 24.02),
    ("taylorseer",                     2.04, 20.56),
    ("taylorseer",                     2.83, 17.40),
    ("teacache",                  1.44, 22.38),
    ("teacache",                  1.73, 22.11),
    ("teacache",                  2.24, 19.28),
    ("sortblock", 1.66, 35.5639),
    ("sortblock", 1.79, 33.9615),
    ("sortblock", 2.00, 31.4957),
    ("sortblock", 2.21, 27.9479),
    # ("sortblockcache-fit-taylor", 2.88, 0.84326),
    ("sortblock", 3.06, 23.3634),
]
# --------------------------

# 鏃犲浘褰㈢晫闈㈡椂鍒囨崲鍒伴潪浜や簰鍚庣
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

# 鎸夋ā鍨嬪垎缁�
groups = {}
for full_name, spd, psnr in records:
    base = full_name.split()[0]          # 鍙栫┖鏍煎墠閮ㄥ垎浣滀负鈥滄ā鍨嬪悕鈥�
    groups.setdefault(base, []).append((spd, psnr, full_name))

fig, ax = plt.subplots(figsize=(7, 4))

for base, pts in groups.items():
    pts.sort(key=lambda x: x[0])         # 鎸夊姞閫熸瘮鍗囧簭锛屼究浜庣敾绾�
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    scatter = ax.scatter(xs, ys, label=base)     # 鏁ｇ偣
    if len(pts) > 1:
        ax.plot(xs, ys, color=scatter.get_facecolors()[0])  # 杩炵嚎

    # for x, y, lbl in pts:                         # 缁欐瘡鐐瑰姞瀹屾暣鏍囩
    #     ax.text(x, y, lbl, fontsize=8, ha="right", va="bottom")

ax.set_xlabel("Speed-up")
ax.set_ylabel("PSNR")
ax.set_title("Speed-up vs. Image Quality (PSNR)")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Model", fontsize=7)
plt.tight_layout()

# 鏈� / 鏃� GUI 鐨勪袱绉嶈緭鍑烘柟寮�
if matplotlib.get_backend() == "Agg":
    out_file = "accel_vs_psnr.png"
    plt.savefig(out_file, dpi=150)
    print(f"鏃犲浘褰㈢晫闈細鍥惧凡淇濆瓨涓� {out_file}")
else:
    plt.show()