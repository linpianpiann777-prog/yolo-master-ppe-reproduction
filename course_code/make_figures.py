"""Step 5: build every report figure from the saved results (no GPU needed).

Outputs print-ready PNG/JPG (white background, 200 dpi) to report/figures/. Figures whose input
results do not exist yet are skipped with a message.

Usage:
    python course_code/make_figures.py
"""

import ast
import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"
PPE = ROOT / "datasets" / "construction-ppe"
RES = ROOT / "results"
RUNS = RES / "ppe"
NAMES = ["helmet", "gloves", "vest", "boots", "goggles", "none", "Person", "no_helmet", "no_goggle", "no_gloves", "no_boots"]
ZH = {
    "helmet": "安全帽", "gloves": "手套", "vest": "反光背心", "boots": "安全靴", "goggles": "护目镜",
    "none": "无背心", "Person": "人员", "no_helmet": "未戴安全帽", "no_goggle": "未戴护目镜",
    "no_gloves": "未戴手套", "no_boots": "未穿安全靴",
}
GROUP = {**{n: "防护装备" for n in ["helmet", "gloves", "vest", "boots", "goggles"]}, "Person": "人员",
         **{n: "违规/缺失" for n in ["none", "no_helmet", "no_goggle", "no_gloves", "no_boots"]}}
# Validated categorical palette (slots 1-5) and ink colours.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b3"
RUN_LABEL = {"YOLO-Master-v0.1-N": "官方默认微调", "YOLO-Master-v0.1-N-clspw0.5": "类别加权 (cls_pw=0.5)"}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "legend.frameon": False,
})


def title(ax, s):
    ax.set_title(s, fontsize=10.5, color=INK, loc="left")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("  ", name)


def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


# ------------------------------------------------------------------ official numbers (paper Table 1/5/6/7)
OFFICIAL_MODELS = ["YOLOv10-N", "YOLOv11-N", "YOLOv12-N", "YOLOv13-N", "YOLO-Master-N"]
OFFICIAL_DATA = {  # mAP per dataset as published in the paper's Table 1
    "COCO": [38.5, 39.4, 40.6, 41.6, 42.4], "VOC": [60.6, 61.0, 60.7, 60.7, 62.1],
    "VisDrone": [18.7, 18.5, 18.3, 17.5, 19.6], "KITTI": [66.0, 67.8, 67.6, 67.7, 69.2],
    "SKU-110K": [57.4, 57.4, 57.4, 57.5, 58.2],
}
OFFICIAL_LAT = [1.84, 1.50, 1.64, 1.97, 1.62]


def fig_official_generalization():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ds = list(OFFICIAL_DATA)
    x, w = np.arange(len(ds)), 0.16
    for i, m in enumerate(OFFICIAL_MODELS):
        ax.bar(x + (i - 2) * w, [OFFICIAL_DATA[d][i] for d in ds], w * 0.92, color=C[0] if i == 4 else GRAY,
               alpha=1 if i == 4 else 0.35 + 0.15 * i, label=m)
    for j, d in enumerate(ds):
        ax.text(x[j] + 2 * w, OFFICIAL_DATA[d][4] + 1, f"{OFFICIAL_DATA[d][4]}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ds)
    ax.set_ylabel("mAP (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    title(ax, "官方结果：Nano 模型在五个数据集上的 mAP（泛化能力）")
    save(fig, "fig_official_generalization.png")


def fig_official_tradeoff():
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    for i, m in enumerate(OFFICIAL_MODELS):
        hl = i == 4
        ax.scatter(OFFICIAL_LAT[i], OFFICIAL_DATA["COCO"][i], s=70 if hl else 45, color=C[0] if hl else GRAY,
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(m, (OFFICIAL_LAT[i], OFFICIAL_DATA["COCO"][i]), xytext=(6, -3 if i != 1 else 5),
                    textcoords="offset points", fontsize=8.5, color=INK if hl else INK2, weight="bold" if hl else None)
    ax.set_xlabel("延迟 (ms，官方测试硬件)")
    ax.set_ylabel("COCO mAP50-95 (%)")
    ax.set_xlim(1.4, 2.15)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
    title(ax, "官方结果：精度—延迟权衡（越靠左上越好）")
    save(fig, "fig_official_tradeoff.png")


def fig_official_ablation():
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    panels = [
        ("插入位置", ["基线", "颈部", "主干", "主干+颈部"], [60.8, 58.2, 62.1, 54.9], 2),
        ("专家数量 E", ["2", "4", "8"], [61.0, 62.3, 62.0], 1),
        ("激活专家数 K（E=4）", ["1", "2", "3", "4"], [61.3, 61.8, 61.8, 61.9], 1),
    ]
    for ax, (t, xs, ys, best) in zip(axs, panels):
        ax.bar(xs, ys, color=[C[0] if i == best else GRAY for i in range(len(xs))], width=0.6)
        for i, y in enumerate(ys):
            ax.text(i, y + 0.15, f"{y}", ha="center", fontsize=8, color=INK2)
        ax.set_ylim(54, 63.5)
        ax.set_title(t, fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("VOC mAP (%)")
    fig.suptitle("官方消融实验（原论文表5—表7，蓝色为官方选用配置）", x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_official_ablation.png")


# ------------------------------------------------------------------ COCO reproduction
def fig_coco_compare():
    rows = {r["model"]: r for r in read_csv(RES / "coco" / "summary.csv")}
    if not rows:
        return print("  skip coco compare")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    series = [("原论文公布", C[3]), ("官方仓库公布", C[1]), ("本地复现", C[0])]
    vals = {
        "AP": {"EsMoE-N": [42.4, 42.7, float(rows["YOLO-Master-EsMoE-N"]["AP"])],
               "v0.1-N": [None, 42.9, float(rows["YOLO-Master-v0.1-N"]["AP"])]},
        "AP50": {"EsMoE-N": [59.2, 58.7, float(rows["YOLO-Master-EsMoE-N"]["AP50"])],
                 "v0.1-N": [None, 59.2, float(rows["YOLO-Master-v0.1-N"]["AP50"])]},
    }
    for ax, metric in zip(axs, ["AP", "AP50"]):
        models = list(vals[metric])
        x, w = np.arange(len(models)), 0.26
        for i, (lab, col) in enumerate(series):
            ys = [vals[metric][m][i] for m in models]
            xs = [x[j] + (i - 1) * w for j in range(len(models)) if ys[j] is not None]
            yv = [y for y in ys if y is not None]
            ax.bar(xs, yv, w * 0.92, color=col, label=lab)
            for xx, yy in zip(xs, yv):
                ax.text(xx, yy + 0.15, f"{yy:.1f}", ha="center", fontsize=7.5, color=INK2)
        lo = min(v for m in models for v in vals[metric][m] if v is not None)
        ax.set_ylim(lo - 2, lo + 3.2)
        ax.set_xticks(x, models)
        ax.set_title(f"COCO val2017 {metric} (%)", fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("官方公布指标与本地复现指标对比", x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_coco_official_vs_local.png")

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    x, w = np.arange(3), 0.34
    for i, (m, col) in enumerate([("YOLO-Master-EsMoE-N", C[0]), ("YOLO-Master-v0.1-N", C[1])]):
        ys = [float(rows[m][k]) for k in ("AP_small", "AP_medium", "AP_large")]
        ax.bar(x + (i - 0.5) * w, ys, w * 0.92, color=col, label=m.replace("YOLO-Master-", ""))
        for xx, yy in zip(x + (i - 0.5) * w, ys):
            ax.text(xx, yy + 0.8, f"{yy:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, ["小目标 (<32²)", "中目标", "大目标 (>96²)"])
    ax.set_ylabel("AP (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    title(ax, "本地复现：不同尺度目标的 AP")
    save(fig, "fig_coco_size_ap.png")


def grid_images(paths, name, cols=2, cell=900):
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        return print(f"  skip {name}")
    rows = -(-len(paths) // cols)
    ims = [Image.open(p).convert("RGB") for p in paths]
    h = max(int(cell * im.height / im.width) for im in ims)
    sheet = Image.new("RGB", (cols * cell, rows * h), "white")
    for i, im in enumerate(ims):
        im = im.resize((cell, int(cell * im.height / im.width)))
        sheet.paste(im, ((i % cols) * cell, (i // cols) * h))
    sheet.save(OUT / name, quality=92)
    print("  ", name)


# ------------------------------------------------------------------ dataset analysis
def load_boxes(split):
    imgs = {p.stem for p in (PPE / "images" / split).iterdir()}
    boxes, per_img = [], []
    for f in (PPE / "labels" / split).glob("*.txt"):
        if f.stem not in imgs:
            continue
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        per_img.append(len(rows))
        boxes += [(int(r[0]), *map(float, r[1:5])) for r in rows]
    return np.array(boxes), per_img


def fig_dataset():
    data = {s: load_boxes(s) for s in ("train", "val", "test")}
    # (1) class share per split
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x, w = np.arange(len(NAMES)), 0.27
    for i, (s, lab) in enumerate([("train", "训练集(样本集)"), ("val", "验证集"), ("test", "测试集")]):
        b = data[s][0]
        share = [100 * (b[:, 0] == k).sum() / len(b) for k in range(len(NAMES))]
        ax.bar(x + (i - 1) * w, share, w * 0.92, color=C[i], label=f"{lab}  {len(b)} 框")
    ax.set_xticks(x, [ZH[n] for n in NAMES], rotation=35, ha="right")
    ax.set_ylabel("占该划分标注框的比例 (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5)
    title(ax, "各类别在训练/验证/测试集中的占比（分布一致性）")
    save(fig, "fig_ds_split_share.png")

    # (2) absolute train counts + images per class
    b = data["train"][0]
    imgs_per_cls = collections.Counter()
    for f in (PPE / "labels" / "train").glob("*.txt"):
        imgs_per_cls.update({int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()})
    order = sorted(range(len(NAMES)), key=lambda k: (b[:, 0] == k).sum())
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(order))
    cnt = [(b[:, 0] == k).sum() for k in order]
    ax.barh(y + 0.2, cnt, 0.38, color=C[0], label="标注框数")
    ax.barh(y - 0.2, [imgs_per_cls[k] for k in order], 0.38, color=C[1], label="出现该类的图像数")
    for yy, v in zip(y, cnt):
        ax.text(v + 15, yy + 0.2, str(v), va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [ZH[NAMES[k]] for k in order])
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    title(ax, "训练集类别样本量（长尾分布）")
    save(fig, "fig_ds_train_counts.png")

    # (3) box size per class (relative sqrt area), boxes per image, centre heatmap
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    order = sorted(range(len(NAMES)), key=lambda k: np.median(np.sqrt(b[b[:, 0] == k][:, 3] * b[b[:, 0] == k][:, 4])))
    sizes = [np.sqrt(b[b[:, 0] == k][:, 3] * b[b[:, 0] == k][:, 4]) * 100 for k in order]
    bp = axs[0].boxplot(sizes, orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True)
    for patch, k in zip(bp["boxes"], order):
        patch.set_facecolor({"防护装备": C[0], "人员": C[2], "违规/缺失": C[1]}[GROUP[NAMES[k]]])
        patch.set_edgecolor("white")
    for med in bp["medians"]:
        med.set_color("white")
    axs[0].set_yticks(range(1, len(order) + 1), [ZH[NAMES[k]] for k in order], fontsize=8)
    axs[0].set_xlabel("相对尺寸 √(w·h) (% 图像边长)")
    axs[0].set_title("各类别目标尺寸", fontsize=9.5, color=INK)
    axs[0].grid(axis="x", color=GRID)
    per = data["train"][1]
    axs[1].hist(per, bins=range(0, max(per) + 2), color=C[0], edgecolor="white")
    axs[1].set_xlabel("每张图像的标注框数")
    axs[1].set_title(f"目标密度（均值 {np.mean(per):.1f}）", fontsize=9.5, color=INK)
    h = axs[2].hist2d(b[:, 1], b[:, 2], bins=24, range=[[0, 1], [0, 1]], cmap="Blues")
    axs[2].invert_yaxis()
    axs[2].set_aspect("equal")
    axs[2].set_xlabel("x 中心")
    axs[2].set_ylabel("y 中心")
    axs[2].set_title("标注框中心热力图", fontsize=9.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_ds_box_stats.png")

    # (4) ground-truth samples
    stems = ["image611", "image1037", "image1354", "image999"]
    font = ImageFont.truetype("msyh.ttc", 18)
    col = {"防护装备": C[0], "人员": C[2], "违规/缺失": C[1]}
    tiles = []
    for s in stems:
        p = next((q for sp in ("train", "val", "test") for q in (PPE / "images" / sp).glob(f"{s}.*")), None)
        lab = next((PPE / "labels" / sp / f"{s}.txt" for sp in ("train", "val", "test") if (PPE / "labels" / sp / f"{s}.txt").exists()), None)
        if p is None:
            continue
        im = Image.open(p).convert("RGB")
        im.thumbnail((560, 560))
        d = ImageDraw.Draw(im)
        for l in lab.read_text().splitlines():
            k, cx, cy, bw, bh = int(l.split()[0]), *map(float, l.split()[1:5])
            x0, y0 = (cx - bw / 2) * im.width, (cy - bh / 2) * im.height
            x1, y1 = (cx + bw / 2) * im.width, (cy + bh / 2) * im.height
            c = col[GROUP[NAMES[k]]]
            d.rectangle([x0, y0, x1, y1], outline=c, width=3)
            d.text((x0 + 2, y0 + 1), ZH[NAMES[k]], fill=c, font=font)
        tiles.append(im)
    W = 560
    sheet = Image.new("RGB", (2 * W + 10, 2 * W + 10), "white")
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 2) * (W + 10) + (W - im.width) // 2, (i // 2) * (W + 10) + (W - im.height) // 2))
    sheet.save(OUT / "fig_ds_gt_samples.jpg", quality=92)
    print("   fig_ds_gt_samples.jpg")


# ------------------------------------------------------------------ PPE results / ablations
def best_val(run):
    rows = read_csv(RUNS / run / "results.csv")
    return max((float(r["metrics/mAP50-95(B)"]) for r in rows), default=-1)


TIE_MARGIN = 0.005  # 0.5 mAP points ~ late-epoch val fluctuation (std 0.39); smaller gains count as a tie


def final_run():
    """Scenario model chosen on the validation split (never the test split); ties keep the simpler official default."""
    base, weighted = "YOLO-Master-v0.1-N", "YOLO-Master-v0.1-N-clspw0.5"
    return weighted if best_val(weighted) > best_val(base) + TIE_MARGIN else base


def read_test():
    return {r["model"]: r for r in read_csv(RUNS / "test_summary.csv")}


def fig_test_ap(rows):
    runs = [m for m in RUN_LABEL if m in rows]
    if not runs:
        return print("  skip test ap")
    base = rows[runs[0]]
    order = sorted(NAMES, key=lambda n: float(base[f"AP50_{n}"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    h = 0.8 / len(runs)
    for i, m in enumerate(runs):
        ys = [y + (i - (len(runs) - 1) / 2) * h for y in range(len(order))]
        ax.barh(ys, [float(rows[m][f"AP50_{n}"]) for n in order], height=h * 0.9, color=C[i], label=RUN_LABEL[m])
    ax.set_yticks(range(len(order)), [ZH[n] for n in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("测试集 AP50 (%)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    title(ax, "测试集各类别 AP50：官方默认微调 vs 类别加权")
    save(fig, "fig_test_ap50_per_class.png")


def fig_curves():
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, m in enumerate(RUN_LABEL):
        rows = read_csv(RUNS / m / "results.csv")
        if rows:
            ax.plot([int(r["epoch"]) for r in rows], [float(r["metrics/mAP50-95(B)"]) * 100 for r in rows],
                    color=C[i], linewidth=1.8, label=RUN_LABEL[m])
    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证集 mAP50-95 (%)")
    ax.grid(axis="y", color=GRID)
    ax.legend(loc="lower right")
    title(ax, "验证集 mAP50-95 随训练轮次变化")
    save(fig, "fig_val_map_curve.png")


def fig_routing():
    rows = read_csv(RES / "ablation" / "routing.csv")
    if not rows:
        return print("  skip routing")
    cases = list(dict.fromkeys(r["case"] for r in rows))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x, w = np.arange(len(cases)), 0.34
    for i, (mode, col, lab) in enumerate([("sparse", C[1], "稀疏路由（官方默认）"), ("dense", C[0], "稠密路由（与训练一致）")]):
        sel = [next(r for r in rows if r["case"] == c and r["mode"].startswith(mode)) for c in cases]
        for ax, key in zip(axs, ["mAP50-95", "inference_ms"]):
            ys = [float(r[key]) for r in sel]
            ax.bar(x + (i - 0.5) * w, ys, w * 0.92, color=col, label=lab)
            for xx, yy in zip(x + (i - 0.5) * w, ys):
                ax.text(xx, yy * 1.02 + 0.1, f"{yy:.1f}", ha="center", fontsize=8, color=INK2)
    labels = ["COCO 官方权重" if "COCO" in c else "PPE 微调模型" for c in cases]
    for ax, t in zip(axs, ["mAP50-95 (%)", "单张推理时间 (ms, RTX 4060)"]):
        ax.set_xticks(x, labels)
        ax.set_title(t, fontsize=9.5, color=INK)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    handles, labels_ = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("消融：ES-MoE 推理路由方式", x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_ablation_routing.png")


def fig_alarm():
    p = RES / "alarm" / "curves.json"
    if not p.exists():
        return print("  skip alarm")
    curves = json.loads(p.read_text(encoding="utf-8"))
    final = final_run()
    # ROC: overall (both models) + per class (final)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for i, m in enumerate(RUN_LABEL):
        if m in curves:
            c = curves[m]["any"]
            axs[0].plot(c["fpr"], c["tpr"], color=C[i], linewidth=1.8, label=f"{RUN_LABEL[m]}  AUC={c['auc']:.3f}")
    for i, (n, c) in enumerate(curves[final]["per_class"].items()):
        axs[1].plot(c["fpr"], c["tpr"], color=C[i], linewidth=1.6, label=f"{ZH[n]}  AUC={c['auc']:.3f}")
    for ax, t in zip(axs, ["图像级违规告警 ROC", "各违规类别 ROC（最终模型）"]):
        ax.plot([0, 1], [0, 1], "--", color=GRAY, linewidth=1)
        ax.set_xlabel("假阳性率 FPR")
        ax.set_ylabel("真阳性率 TPR")
        ax.set_title(t, fontsize=9.5, color=INK)
        ax.legend(loc="lower right", fontsize=7.5)
        ax.grid(color=GRID)
        ax.set_aspect("equal")
    fig.tight_layout()
    save(fig, "fig_alarm_roc.png")

    # threshold sweep on test for the final model, with val-chosen threshold marked
    sw = curves[final]["threshold_sweep"]
    t = np.array(sw["t"])
    P, R, F = np.array(sw["test"]).T * 100
    best = t[int(np.argmax(sw["val_f1"]))]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for y, col, lab in ((P, C[0], "精确率"), (R, C[1], "召回率"), (F, C[2], "F1")):
        ax.plot(t, y, color=col, linewidth=1.8, label=lab)
    ax.axvline(0.25, color=GRAY, linestyle="--", linewidth=1)
    ax.text(0.255, 4, "官方默认 0.25", fontsize=8, color=INK2)
    ax.axvline(best, color=INK, linestyle=":", linewidth=1.2)
    ax.text(best + 0.005, 12, f"验证集选定 {best:.2f}", fontsize=8, color=INK)
    ax.set_xlabel("告警置信度阈值")
    ax.set_ylabel("测试集 (%)")
    ax.set_ylim(0, 102)
    ax.grid(color=GRID)
    ax.legend(loc="center right", fontsize=8)
    title(ax, "告警阈值对精确率/召回率/F1 的影响（测试集）")
    save(fig, "fig_alarm_threshold.png")


def fig_improvement():
    """Official default scheme vs ours on the scenario-level metrics."""
    test, alarm = read_test(), read_csv(RES / "alarm" / "alarm_summary.csv")
    base, ours = "YOLO-Master-v0.1-N", final_run()
    if base not in test or ours not in test or not alarm:
        return print("  skip improvement")
    a = {(r["model"], r["threshold_rule"]): r for r in alarm}
    keys = [("precision", "告警精确率"), ("recall", "告警召回率"), ("F1", "告警 F1"), ("AUC", "告警 AUC")]
    metrics = [k[1] for k in keys]
    off = [float(a[(base, "default conf 0.25")][k]) for k, _ in keys]
    new = [float(a[(ours, "val-tuned")][k]) for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    x, w = np.arange(len(metrics)), 0.36
    ax.bar(x - w / 2, off, w * 0.92, color=GRAY, label="官方默认方案")
    ax.bar(x + w / 2, new, w * 0.92, color=C[0], label="本文方案")
    for i in range(len(metrics)):
        ax.text(x[i] - w / 2, off[i] + 1, f"{off[i]:.1f}", ha="center", fontsize=8, color=INK2)
        ax.text(x[i] + w / 2, new[i] + 1, f"{new[i]:.1f}", ha="center", fontsize=8, color=INK)
        d = new[i] - off[i]
        ax.text(x[i], max(off[i], new[i]) + 5, f"{d:+.1f}", ha="center", fontsize=8.5, color=C[2] if d >= 0 else C[1], weight="bold")
    ax.set_xticks(x, metrics)
    ax.set_ylim(60, 112)
    ax.set_ylabel("测试集 (%)")
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, ncol=2)
    title(ax, "改进效果：官方默认告警（阈值 0.25）vs 本文告警（验证集选定阈值）")
    save(fig, "fig_improvement.png")


def fig_demo_grid(stems, out_name, cell=520):
    """2x2 grid of annotated demo predictions (results/demo, produced by ppe_demo.py)."""
    if not (RES / "demo").exists():
        return print(f"  skip {out_name}")
    sheet = Image.new("RGB", (cell * 2 + 10, cell * 2 + 10), "white")
    for i, s in enumerate(stems):
        im = Image.open(RES / "demo" / f"image{s}.jpg").convert("RGB")
        im.thumbnail((cell, cell))
        sheet.paste(im, ((i % 2) * (cell + 10) + (cell - im.width) // 2, (i // 2) * (cell + 10) + (cell - im.height) // 2))
    sheet.save(OUT / out_name, quality=92)
    print("  ", out_name)


# ------------------------------------------------------------------ code screenshots
def code_figure(file, func, out_name):
    from pygments import highlight
    from pygments.formatters import ImageFormatter
    from pygments.lexers import PythonLexer

    src = (ROOT / "course_code" / file).read_text(encoding="utf-8")
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == func)
    lines = src.splitlines()[node.lineno - 1 - len(node.decorator_list): node.end_lineno]
    indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
    code = "\n".join(l[indent:] for l in lines)
    png = highlight(code, PythonLexer(), ImageFormatter(font_name="Consolas", font_size=15, line_numbers=True,
                                                         line_number_bg="#f0efec", line_number_fg="#8a8984",
                                                         style="default", image_pad=14, line_pad=3))
    (OUT / out_name).write_bytes(png)
    print("  ", out_name)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_official_generalization()
    fig_official_tradeoff()
    fig_official_ablation()
    fig_coco_compare()
    grid_images([RES / "coco" / "YOLO-Master-v0.1-N" / f for f in ("BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png", "BoxPR_curve.png")],
                "fig_coco_metric_curves.jpg")
    fig_dataset()
    rows = read_test()
    fig_test_ap(rows)
    fig_curves()
    final = final_run()
    grid_images([RUNS / f"{final}_test" / f for f in ("BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png", "BoxPR_curve.png")],
                "fig_ppe_metric_curves.jpg")
    fig_routing()
    fig_alarm()
    fig_improvement()
    fig_demo_grid(["611", "1037", "772", "999"], "fig_demo_compliant.jpg")
    fig_demo_grid(["1354", "1231", "1145", "1132"], "fig_demo_alerts.jpg")
    code_figure("train_ppe.py", "dense_esmoe", "fig_code_dense_routing.png")
    code_figure("alarm_analysis.py", "image_scores", "fig_code_alarm_scores.png")
    code_figure("ppe_demo.py", "find_violations", "fig_code_demo.png")
    print(f"figures saved to {OUT}")
