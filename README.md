# 深度学习在施工现场个人防护装备检测场景的应用

复现截图高亮论文 **YOLO-Master: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection**，并迁移到施工现场 PPE 检测。提供课程代码、真实训练记录、模型权重、中文 Word 报告与打印 PDF，完整训练/验证/测试图像及标注通过本仓库 Release 提供。

## 实际完成范围

使用官方发布权重，在 COCO val2017 全部 5000 张图像上评估，复核保存的全量预测并用 pycocotools 重新计算 AP。**没有从头训练 COCO 600 轮，也没有复现论文 TensorRT 1.62 ms 延迟**。本项目属于官方权重评估复现与新场景迁移训练，不能将其写成完整从头训练复现。高难度作业是否接受这种复现范围，应由教师评定。

训练复用本机 2026-11-02 至 11-03 已完成的实验，2026-11-12 重新核验数据与指标，并在 RTX 4060 Laptop 8GB 上重新运行全部三个模型的独立 PPE 测试。未将旧训练表述为本次新训练。原始日志在 `logs/`，本次复核在 `verification/`。

| 评估对象 | 数据与协议 | AP50–95 | AP50 |
|---|---|---:|---:|
| 官方 EsMoE-N 权重 | COCO val2017，pycocotools | 42.8% | 59.3% |
| 官方 v0.1-N 权重 | COCO val2017，pycocotools | 43.1% | 59.7% |
| v0.1-N 迁移 100 轮 | PPE 独立 test，Ultralytics | 27.6% | 56.8% |
| 类别加权 v0.1-N 100 轮 | PPE 独立 test，Ultralytics | 26.9% | 54.0% |
| EsMoE-N 迁移 50 轮 | PPE 独立 test，Ultralytics | 27.6% | 55.9% |

论文摘要报告 COCO AP 42.4%，官方当前模型库分别报告 EsMoE-N 42.7%、v0.1-N 42.9%。权重版本与评估实现存在差异，达到接近的精度不能证明训练过程完全复现。COCO 与 PPE 的类别、数据和评估器均不同，不可横向比较其 AP。类别加权在本实验中未带来整体提升；没有修改或美化结果。

三个正式训练 CSV 累计耗时分别为 **3312.72、3349.32、1964.18 秒**，合计 **143.77 分钟**。该统计不含数据下载、环境安装、失败尝试、开发和报告制作。重训练脚本设有总计四小时进程截止控制；硬件速度变化时会提前停止，不能保证届时训练完全部轮数或达到既有指标。

## 文件与下载

- [课程报告 Word](report/YOLO-Master_PPE_Course_Design.docx)；[打印版 PDF](report/YOLO-Master_PPE_Course_Design.pdf)。封面已按所提供课程模板填写：24级、智科一班、刘炘扬、学号2404120128。
- [数据与实验数据 Release](https://github.com/linpianpiann777-prog/yolo-master-ppe-reproduction/releases/tag/v1.0)：`construction-ppe-dataset.zip` 含全部 1416 张图像与原始标注；`coco-predictions-evidence.zip` 含 COCO 全量预测、路由消融与复核证据。
- `course_code/`：训练、测试、COCO 官方评估、路由消融、告警分析、预测演示、数据下载及校验。
- `YOLO-Master/`：官方实现中的完整 `ultralytics/` 包及构建/许可文件，固定上游 commit `c73e53716bf13fa5344c66f7b77cde2ffde051fd`，非完整上游开发仓库；原始来源 https://github.com/Tencent/YOLO-Master 。
- `weights/`：两份官方 Nano 预训练权重；`results/ppe/*/weights/best.pt`：三个实训最佳权重。为减小体积未分发训练 `last.pt`，因此新电脑需重新训练，不能从课程实验最后一步恢复优化器。
- `verification/dataset_manifest.csv`：每张图像和标签的 SHA256；`audit.json`：真实样本/框数、跨集合完全重复检测与训练时长；`coco_recheck.json`：本次重新计算的 COCO 指标；`ppe_recheck.log`：本次重新运行的独立测试。

## 数据来源与边界

Construction-PPE 源自 [Ultralytics 官方公开数据集](https://docs.ultralytics.com/datasets/detect/construction-ppe/)，原始包 https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip ，含原始 AGPL-3.0 许可，未生成虚假图像或合成指标。保留官方 train 1132 / val 143 / test 141 划分，11 类。模型选择与告警阈值使用 val，最终指标使用独立 test。

逐张图片可读性、标签格式、类别及归一化坐标均通过检查，集合之间 SHA256 完全重复数为零。训练标签目录存在 10 个没有对应图像的原始标签，保留并在审计中列出，不计入训练框数。没有完全重复图片不等于没有相似帧或相同工地，尚未验证人员、视频、站点级隔离。缺失装备类别较少，测试定位精度较低，图像级告警不能代替逐人违规判断。

COCO 标注和图像来自 [COCO 官方](https://cocodataset.org/#download)。为减少重复分发，COCO 的 5000 张验证图像由官方地址下载，保存的全量预测已上传本仓库 Release。COCO 标注采用 CC BY 4.0，图像保留各自原始许可。下载链接：

- https://images.cocodataset.org/zips/val2017.zip
- https://images.cocodataset.org/annotations/annotations_trainval2017.zip

放置到 `datasets/coco/images/val2017/` 和 `datasets/coco/annotations/instances_val2017.json`。创建 `datasets/coco/val2017.txt`，每行一个验证图像绝对路径。PPE 数据单独放置在 `datasets/construction-ppe/`。

## 本机运行

本机训练环境可复用：`E:\yolo_course\.venv\Scripts\python.exe`。项目路径为 `E:\deeplearn\yolo-master-ppe`。PyCharm 选择该解释器，工作目录选择本项目，然后运行 `course_code/ppe_demo.py`。默认识别真实独立测试集并保存标注图与告警日志。

```powershell
cd E:\deeplearn\yolo-master-ppe
$env:PYTHONPATH="$PWD\YOLO-Master"
& E:\yolo_course\.venv\Scripts\python.exe course_code\ppe_demo.py
```

## 其他电脑安装与复核

需要 Git、Python 3.12，GPU 重训练需要 CUDA 版 PyTorch。本项目依赖 YOLO-Master 分支，**不可用普通 PyPI ultralytics 替代**。

```powershell
git clone https://github.com/linpianpiann777-prog/yolo-master-ppe-reproduction.git
cd yolo-master-ppe-reproduction
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r course_code\requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -e YOLO-Master
.venv\Scripts\python course_code\download_data.py
.venv\Scripts\python course_code\audit_delivery.py
.venv\Scripts\python course_code\test_ppe.py
.venv\Scripts\python course_code\ppe_demo.py
```

`download_data.py` 从本仓库 Release 下载并核对 SHA256，不会生成占位数据。数据配置运行时自动改为当前项目绝对路径。独立测试重跑会覆盖对应测试输出；历史指标和日志另见 Git 记录与 Release 证据。本机实测 PyTorch 2.11.0+cu128、Python 3.12.10、RTX 4060 Laptop 8GB，完整环境清单见锁定文件。

重训练三组实验并执行共享四小时截止控制：

```powershell
.venv\Scripts\python course_code\run_training_budget.py
```

默认结果写入名称带 `-retrain` 的新目录，保留交付权重。设置每组上限 0.95、0.95、0.65 小时，总进程到四小时停止。超时可能发生在一个 epoch 中途，以已有检查点为准。原实验还尝试稀疏/稠密路由、验证集阈值选择；查看 `ablation_routing.py`、`alarm_analysis.py` 和报告。推理延迟受硬件、预热、批次影响，单次测量不能承诺部署加速。

COCO 验证图像准备完成后：

```powershell
.venv\Scripts\python course_code\eval_coco.py
```

如只重算已保存预测的 AP，先解压 Release 的 COCO 证据包，再执行 `course_code/recheck_coco_saved.py`，仍需官方标注。

## 报告与许可

报告包含场景、原理、环境、指标对比、改进与负结果、路由消融、告警评价、限制和参考文献。打印 PDF 为 Word 导出，电子稿可编辑。重建需 Node.js 的 `docx` 包并运行 `node report/build_report.js`；Windows Word 可使用 `report/export_pdf.ps1 -PythonExe <含lxml的Python解释器>` 更新目录、统一排版和交叉引用后导出 PDF。`report/final_format_report.py` 将正文设为宋体黑色，主标题14磅、子标题12磅、正文10.5磅，首行缩进两字符；数据表为三线表，图表和章节引用支持内部跳转。封面按课程范例设置，首页题目42磅、课程信息14磅，封面填写文字与标签底部对齐、贴近横线，目录按三级缩进并保留50个跳转条目。摘要与目录使用罗马页码，正文从1重新编号。

本项目遵循 **AGPL-3.0**；上游源码、权重和数据版权归原作者。根目录 `LICENSE` 及 `THIRD_PARTY.md` 列明来源。不声称论文原始架构或公开数据集为本人原创，不声称少数类检测已达到实际安全生产部署要求。
