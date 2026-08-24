# Phase 1 — 基础重构 Implementation Plan

> **For agentic workers:**
> **状态：已完成（2026-08）** — 本 Phase 1 全部步骤已在代码中实现并通过测试，后续 Phase 2/3/4 也已落地，见 spec `2026-05-03-meteor-detection-redesign.md` §10。
> 该文件 checkbox 已全部勾选。
 REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 修复现有 Bug，搭建新目录结构，将现有代码提取为 `core/` 引擎，配置迁移到 YAML，增加 logging。

**Architecture:** `core/` 作为纯业务逻辑层，不依赖任何 Web 框架。现有 `src/` 中的代码按职责拆分到 `core/preprocessing/`、`core/detection/`、`core/utils/`。配置从 Python 模块改为 YAML 文件 + 运行时验证。

**Tech Stack:** Python 3.12, OpenCV, NumPy, PyYAML, logging（标准库）

---

## File Map

### 新建文件

| 文件 | 职责 |
|------|------|
| `config.yaml` | 统一配置（路径、算法参数） |
| `core/__init__.py` | core 包初始化，导出主要接口 |
| `core/pipeline.py` | 检测流水线编排 |
| `core/preprocessing/__init__.py` | preprocessing 子包 |
| `core/preprocessing/fisheye.py` | 鱼眼校正（从 `src/rect_fisheye.py` 迁移） |
| `core/preprocessing/merge.py` | 图像叠加（从 `src/merge_pics.py` 迁移） |
| `core/preprocessing/background.py` | 背景差分/去噪（从 `src/image_identify.py` 提取） |
| `core/detection/__init__.py` | detection 子包 |
| `core/detection/line_detector.py` | 直线检测 + 验证（从 `src/image_identify.py` 重构） |
| `core/detection/object_detector.py` | AI 模型检测（从 `src/detect_image.py` 迁移） |
| `core/detection/tracker.py` | 轨迹跟踪（从 `src/meteor_tracker.py` 迁移） |
| `core/utils/__init__.py` | utils 子包 |
| `core/utils/config.py` | 配置加载器（读 YAML，运行时验证） |
| `core/utils/image_io.py` | 图像读写工具 |
| `core/utils/logging.py` | 日志配置 |
| `cli/meteo.py` | CLI 入口（替代根目录的 main.py/test.py） |
| `tests/conftest.py` | pytest fixtures |
| `tests/test_preprocessing/test_fisheye.py` | 鱼眼校正测试 |
| `tests/test_preprocessing/test_background.py` | 背景差分测试 |
| `tests/test_detection/test_line_detector.py` | 直线检测测试 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | 添加 PyYAML 依赖，添加 dev 依赖（pytest） |
| `.gitignore` | 添加 storage/、logs/ |

### 删除/清理文件

| 文件 | 原因 |
|------|------|
| `config.py` | 被 config.yaml + core/utils/config.py 替代 |
| `main.py` | 被 cli/meteo.py 替代 |
| `test.py` | 被正式测试替代 |
| `src/` 目录下所有文件 | 已迁移到 core/ |

---

## Task 1: 创建新目录结构和 pyproject.toml 更新

**Files:**
- Modify: `pyproject.toml`
- Create: `core/__init__.py`, `core/preprocessing/__init__.py`, `core/detection/__init__.py`, `core/utils/__init__.py`
- Create: `cli/__init__.py`
- Create: `tests/__init__.py`, `tests/test_preprocessing/__init__.py`, `tests/test_detection/__init__.py`

- [x] **Step 1: 创建目录结构**

```bash
mkdir -p core/preprocessing core/detection core/utils cli tests/test_preprocessing tests/test_detection storage/uploads storage/results storage/logs
```

- [x] **Step 2: 创建所有 __init__.py**

```bash
touch core/__init__.py core/preprocessing/__init__.py core/detection/__init__.py core/utils/__init__.py cli/__init__.py tests/__init__.py tests/test_preprocessing/__init__.py tests/test_detection/__init__.py
```

- [x] **Step 3: 更新 pyproject.toml 添加新依赖**

在 `[tool.poetry.dependencies]` 部分添加：
```toml
pyyaml = "^6.0"
```

在文件末尾添加：
```toml
[tool.poetry.group.dev.dependencies]
pytest = "^8.0"
```

- [x] **Step 4: 安装依赖**

Run: `cd /Users/jiangc/python_workspace/meteo-detection && poetry install`
Expected: 成功安装 pyyaml 和 pytest

- [x] **Step 5: 验证 pytest 可用**

Run: `poetry run pytest --version`
Expected: pytest 8.x.x

- [x] **Step 6: Commit**

```bash
git add pyproject.toml core/ cli/ tests/ poetry.lock
git commit -m "chore: create project structure and add dependencies"
```

---

## Task 2: 配置迁移 — config.yaml

**Files:**
- Create: `config.yaml`
- Create: `core/utils/config.py`

- [x] **Step 1: 创建 config.yaml**

内容：项目所有配置项，包括路径（用相对路径）、算法参数（Hough、流星检测、轨迹跟踪、背景建模）。路径使用 `~` 或相对路径，运行时由配置加载器解析为绝对路径。

- [x] **Step 2: 编写 core/utils/config.py 配置加载器**

职责：
  - 从 `config.yaml` 加载配置
  - 路径自动转为绝对路径（基于项目根目录）
  - 提供类型安全的访问函数（`get_path()`, `get_algorithm_params()` 等）
  - 配置缺失或无效时抛出明确的错误
  - 支持环境变量覆盖（格式：`METEO__SECTION__KEY=value`）

- [x] **Step 3: 编写配置加载器的基本验证**

Run: `poetry run python -c "from core.utils.config import load_config; c = load_config(); print(c)"`
Expected: 打印完整配置，路径为绝对路径，无异常

- [x] **Step 4: Commit**

```bash
git add config.yaml core/utils/config.py
git commit -m "feat: migrate config from Python module to YAML with validation"
```

---

## Task 3: Logging 基础设施

**Files:**
- Create: `core/utils/logging.py`

- [x] **Step 1: 编写日志配置模块**

职责：
  - 配置 root logger，同时输出到控制台和文件
  - 文件 handler：写入 `storage/logs/`，按天轮转
  - 格式：`%(asctime)s [%(levelname)s] %(name)s: %(message)s`
  - 提供 `get_logger(name)` 函数供各模块使用
  - DEBUG 级别写文件，INFO 级别输出控制台

- [x] **Step 2: 验证日志系统**

Run: `poetry run python -c "from core.utils.logging import get_logger; log = get_logger('test'); log.info('hello'); log.debug('world')"`
Expected: 控制台显示 INFO 消息，storage/logs/ 下的日志文件包含 DEBUG 和 INFO 消息

- [x] **Step 3: Commit**

```bash
git add core/utils/logging.py
git commit -m "feat: add logging infrastructure with file rotation"
```

---

## Task 4: 迁移鱼眼校正模块

**Files:**
- Create: `core/preprocessing/fisheye.py`
- Test: `tests/test_preprocessing/test_fisheye.py`

- [x] **Step 1: 将 `src/rect_fisheye.py` 的逻辑迁移到 `core/preprocessing/fisheye.py`**

改动：
  - 保留 `rectify_fisheye()` 函数核心逻辑
  - 参数 `dtype`, `format`, `fov`, `pfov` 改为从 config.yaml 读取默认值，同时允许调用方覆盖
  - 用 logging 替换 print
  - 添加类型注解
  - 添加 docstring

- [x] **Step 2: 编写基本测试**

测试：用 `test/rectify_test/` 中的图片运行校正，验证输出文件存在且尺寸正确。

- [x] **Step 3: 运行测试**

Run: `poetry run pytest tests/test_preprocessing/test_fisheye.py -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add core/preprocessing/fisheye.py tests/test_preprocessing/test_fisheye.py
git commit -m "feat: migrate fisheye correction to core/preprocessing"
```

---

## Task 5: 迁移图像叠加模块

**Files:**
- Create: `core/preprocessing/merge.py`

- [x] **Step 1: 将 `src/merge_pics.py` 的逻辑迁移到 `core/preprocessing/merge.py`**

改动：
  - 保留 `merge_images()` 核心逻辑
  - 添加类型注解和 docstring
  - 用 logging 替换 print
  - 增加输入验证（目录非空、图片格式检查）

- [x] **Step 2: 编写基本测试**

测试：用 `test/merge_tests/` 中的图片运行叠加，验证输出文件存在。

- [x] **Step 3: 运行测试**

Run: `poetry run pytest tests/ -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add core/preprocessing/merge.py
git commit -m "feat: migrate image merging to core/preprocessing"
```

---

## Task 6: 背景差分模块重构

**Files:**
- Create: `core/preprocessing/background.py`

- [x] **Step 1: 从 `src/image_identify.py` 中提取背景差分逻辑**

将 `ImageSelector` 中的以下方法提取为独立函数：
  - `remove_noise_from_light_image()` → `remove_background()`
  - `_remove_noise_with_registration()` → `_remove_background_with_registration()`

新模块提供统一接口：
  - `remove_background(light, background, method='registration')` — 支持多种方法
  - `method='simple'`：简单差分（原有逻辑）
  - `method='registration'`：配准 + 差分（原有逻辑）
  - 后续扩展：`method='mog2'`（Phase 3）

- [x] **Step 2: 用 `test/rectify_test/` 中的图片做集成测试**

Run: `poetry run python -c "from core.preprocessing.background import remove_background; import cv2; ..."`
Expected: 输出去噪后的图像，无异常

- [x] **Step 3: Commit**

```bash
git add core/preprocessing/background.py
git commit -m "feat: extract background subtraction into core/preprocessing"
```

---

## Task 7: 直线检测模块重构（修复核心 Bug）

**Files:**
- Create: `core/detection/line_detector.py`
- Test: `tests/test_detection/test_line_detector.py`

这是最重要的重构任务，修复多个 Bug。

- [x] **Step 1: 从 `src/image_identify.py` 中提取直线检测逻辑，重构为 `core/detection/line_detector.py`**

重构要点：
  - `detect_lines()` → `LineDetector.detect()`：统一返回 `DetectionResult` 列表（包含 x1, y1, x2, y2, brightness, confidence）
  - **修复斜率验证 Bug**：`validate_meteor_line()` 不再使用 ROI 对角线斜率作为预期值。改为评分制：
    - 长度评分（越长分越高）
    - 亮度评分（ROI 内像素均值高于背景越多分越高）
    - 长宽比评分（越狭长分越高）
    - 综合评分 > 阈值 → 判定为候选流星
  - **修复返回值不一致**：统一返回 `List[DetectionResult]`，无结果返回空列表
  - 用 logging 替换所有 print
  - Hough 参数从 config.yaml 读取

- [x] **Step 2: 编写测试**

测试：
  - 用合成图像（画一条白线在黑色背景上）验证检测能检出直线
  - 验证无直线时返回空列表（不是 None）
  - 验证评分机制：亮线得分高，暗线得分低

- [x] **Step 3: 运行测试**

Run: `poetry run pytest tests/test_detection/test_line_detector.py -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add core/detection/line_detector.py tests/test_detection/test_line_detector.py
git commit -m "feat: refactor line detector with scoring-based validation (fixes slope bug)"
```

---

## Task 8: 迁移轨迹跟踪模块

**Files:**
- Create: `core/detection/tracker.py`

- [x] **Step 1: 将 `src/meteor_tracker.py` 迁移到 `core/detection/tracker.py`**

改动：
  - 保留 `MeteorTracker` 类全部逻辑
  - 配置参数从 config.yaml 读取（替代 config.py 的 TRACKING 字典）
  - 用 logging 替换 print
  - 添加类型注解

- [x] **Step 2: 编写基本测试**

测试：模拟 3 帧的直线输入，验证轨迹关联和流星确认逻辑。

- [x] **Step 3: 运行测试**

Run: `poetry run pytest tests/ -v`
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add core/detection/tracker.py
git commit -m "feat: migrate meteor tracker to core/detection"
```

---

## Task 9: 迁移 AI 模型检测模块

**Files:**
- Create: `core/detection/object_detector.py`

- [x] **Step 1: 将 `src/detect_image.py` 迁移到 `core/detection/object_detector.py`**

改动：
  - 保留 `detect_from_image()` 和 `extract_bboxes()` 核心逻辑
  - 模型路径从 config.yaml 读取
  - 用 logging 替换 print
  - 添加类型注解
  - 将 TF 模型加载改为懒加载（首次调用时加载，不在 import 时加载）

- [x] **Step 2: 验证模块可导入**

Run: `poetry run python -c "from core.detection.object_detector import detect_from_image; print('import ok')"`
Expected: 打印 "import ok"，无异常

- [x] **Step 3: Commit**

```bash
git add core/detection/object_detector.py
git commit -m "feat: migrate object detector to core/detection with lazy loading"
```

---

## Task 10: 图像读写工具模块

**Files:**
- Create: `core/utils/image_io.py`

- [x] **Step 1: 编写图像读写工具**

提供统一接口：
  - `load_image(path)` — 支持 JPG/PNG，返回 numpy array
  - `load_raw(path)` — 使用 rawpy 读取 RAW 格式
  - `save_image(img, path)` — 保存图像
  - `is_supported_format(path)` — 检查文件格式是否支持

- [x] **Step 2: Commit**

```bash
git add core/utils/image_io.py
git commit -m "feat: add image I/O utility with RAW support"
```

---

## Task 11: 流水线编排

**Files:**
- Create: `core/pipeline.py`

- [x] **Step 1: 编写检测流水线**

`MeteorPipeline` 类，步骤：
  1. `load_images()` — 从目录加载图片列表
  2. `preprocess()` — 鱼眼校正（可选）→ 背景差分
  3. `detect()` — 直线检测
  4. `track()` — 轨迹跟踪（多帧时）
  5. `output()` — 输出标注图 + CSV + JSON

每个步骤支持回调函数（用于向 Web 层推送进度）。

- [x] **Step 2: 编写端到端冒烟测试**

用 `test/rectify_test/` 中的少量图片跑完整流水线，验证不报错。

- [x] **Step 3: Commit**

```bash
git add core/pipeline.py
git commit -m "feat: add detection pipeline orchestrator"
```

---

## Task 12: CLI 入口

**Files:**
- Create: `cli/meteo.py`
- Modify: `pyproject.toml`（添加 `[tool.poetry.scripts]` 或 `[project.scripts]`）

- [x] **Step 1: 使用 Click 编写 CLI**

命令：
  - `meteo detect-single <image>` — 单张检测
  - `meteo detect-batch <directory>`  - 批量检测
  - `meteo rectify <input> <output>` — 鱼眼校正
  - `meteo merge <directory> <output>` — 图像叠加

每个命令调用 `core/` 中对应的模块。

- [x] **Step 2: 验证 CLI**

Run: `poetry run meteo --help`
Expected: 显示所有子命令和帮助信息

- [x] **Step 3: 测试单张检测**

Run: `poetry run meteo detect-single test/rectify_test/DSC03826.jpg`
Expected: 输出检测结果，无异常

- [x] **Step 4: Commit**

```bash
git add cli/meteo.py pyproject.toml
git commit -m "feat: add CLI entry point with Click"
```

---

## Task 13: 清理旧文件 + .gitignore 更新

**Files:**
- Delete: `config.py`, `main.py`, `test.py`
- Delete: `src/` 目录下所有文件（已迁移完毕）
- Modify: `.gitignore`

- [x] **Step 1: 更新 .gitignore**

添加：
```
storage/
logs/
*.log
__pycache__/
.venv/
```

- [x] **Step 2: 删除旧文件**

```bash
rm config.py main.py test.py
rm -rf src/
```

- [x] **Step 3: 运行全量测试确认无断裂**

Run: `poetry run pytest tests/ -v`
Expected: 所有测试 PASS，无 import 错误

- [x] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: clean up legacy files and update gitignore"
```

---

## Task 14: Phase 1 收尾验证

- [x] **Step 1: 运行全量测试**

Run: `poetry run pytest tests/ -v`
Expected: 全部 PASS

- [x] **Step 2: 运行 CLI 冒烟测试**

Run: `poetry run meteo detect-batch test/rectify_test/`
Expected: 正常处理，输出结果到 storage/results/

- [x] **Step 3: 验证日志**

Run: `ls storage/logs/`
Expected: 存在日志文件，包含本次运行的日志

- [x] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "chore: Phase 1 complete — refactor to core engine with tests"
```
