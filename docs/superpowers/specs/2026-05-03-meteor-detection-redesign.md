# 流星探测项目重新设计文档

> 日期：2026-05-03
> 状态：待实施

---

## 1. 目标

将现有流星探测项目从"散装脚本集合"升级为**可部署的 Web 应用**，支持单张/批量/实时三种检测模式，具备完整的前端管理界面和任务队列。

---

## 2. 架构

### 2.1 整体架构：任务队列模式

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  FastAPI     │────▶│  Redis       │────▶│  Celery Worker  │
│  Web 前端    │     │  任务队列     │     │  检测引擎        │
│  + API       │◀────│              │◀────│  (CPU/GPU密集)   │
└─────────────┘     └──────────────┘     └─────────────────┘
       │                                          │
       ▼                                          ▼
┌─────────────┐                          ┌─────────────────┐
│  SQLite     │                          │  文件系统         │
│  数据库      │                          │  (图片/结果/日志) │
└─────────────┘                          └─────────────────┘
```

### 2.2 技术栈

| 层 | 技术 | 选型理由 |
|---|------|---------|
| Web 框架 | FastAPI + Jinja2 | Python 原生，异步支持好，SSR 够用 |
| 前端交互 | HTMX | 无需构建，局部刷新够用 |
| 任务队列 | Celery + Redis | Python 生态标准方案 |
| 数据库 | SQLite | 轻量单文件，未来换 PostgreSQL |
| 图表 | Chart.js | 轻量，亮度曲线和轨迹可视化 |
| 实时推送 | SSE | 单向推送任务进度，比 WebSocket 简单 |
| 部署 | Docker Compose | 本地开发 + Linux 服务器一致 |
| CLI | Click | Python CLI 标准库 |
| 检测引擎 | OpenCV + NumPy + TensorFlow | 沿用现有技术 |

---

## 3. 项目结构

```
meteo-detection/
├── pyproject.toml
├── README.md
├── config.yaml                    # 统一配置（替代 config.py）
│
├── api/                           # FastAPI Web 层
│   ├── __init__.py
│   ├── main.py                    # FastAPI app 入口
│   ├── routes/
│   │   ├── tasks.py               # 任务 CRUD
│   │   ├── results.py             # 结果查询
│   │   └── upload.py              # 图片上传
│   ├── models/
│   │   ├── schemas.py             # Pydantic 模型
│   │   └── database.py            # DB 连接
│   └── templates/                 # 前端页面
│       ├── base.html
│       ├── dashboard.html         # 任务看板
│       ├── task_detail.html       # 任务详情 + 结果
│       └── upload.html            # 上传页面
│
├── worker/                        # Celery Worker 层
│   ├── __init__.py
│   ├── celery_app.py              # Celery 配置
│   ├── tasks.py                   # 任务定义
│   └── runners/
│       ├── batch.py               # 批量处理任务
│       └── realtime.py            # 实时检测任务
│
├── core/                          # 核心检测引擎（纯逻辑，无框架依赖）
│   ├── __init__.py
│   ├── pipeline.py                # 流水线编排
│   ├── preprocessing/
│   │   ├── fisheye.py             # 鱼眼校正（已有）
│   │   ├── merge.py               # 图像叠加（已有）
│   │   └── background.py          # 背景建模（重构）
│   ├── detection/
│   │   ├── line_detector.py       # 直线检测（重构）
│   │   ├── object_detector.py     # AI 模型检测（已有）
│   │   └── tracker.py             # 轨迹跟踪（已有）
│   ├── classification/
│   │   ├── meteor_classifier.py   # 流星/卫星/飞机分类
│   │   └── satellite_filter.py    # 卫星交叉验证
│   └── utils/
│       ├── exif_reader.py         # EXIF 读取
│       ├── image_io.py            # 图像读写（含 RAW）
│       └── visualization.py       # 结果可视化
│
├── cli/                           # CLI 入口
│   └── meteo.py                   # Click 命令行
│
├── data/                          # 数据文件
│   ├── models/                    # AI 模型
│   └── de421.bsp
│
├── storage/                       # 运行时数据
│   ├── uploads/                   # 上传图片
│   ├── results/                   # 检测结果
│   └── logs/                      # 日志
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
└── tests/
    ├── test_pipeline.py
    ├── test_detection/
    └── test_classification/
```

**核心设计原则**：`core/` 是纯业务逻辑，不依赖 FastAPI 或 Celery，可独立测试和复用。`api/` 和 `worker/` 都只调用 `core/`。

---

## 4. 核心检测流水线

### 4.1 流水线步骤

```
1. Load       → 读取图片序列（支持 JPG/PNG/RAW）
2. Preprocess → 鱼眼校正 → 图像配准 → 背景差分
3. Detect     → 直线检测 + AI 模型检测（可选）
4. Classify   → 流星/卫星/飞机/噪点分类
5. Track      → 多帧轨迹关联
6. Output     → 标注图 + CSV 报告 + JSON 元数据
```

### 4.2 三种运行模式

| 模式 | 输入 | 特点 |
|------|------|------|
| 单张 | 单文件 | 跳过跟踪，只做检测 |
| 批量 | 目录/文件列表 | 完整流水线，批量输出 |
| 实时 | 摄像头/文件夹监听 | 持续运行，滑动窗口跟踪 |

---

## 5. 数据模型

数据库用 SQLite，核心表：

| 表 | 用途 |
|---|------|
| `tasks` | 任务记录（状态、参数、创建时间、完成时间） |
| `images` | 输入图片（路径、EXIF信息、关联任务） |
| `detections` | 单次检测结果（位置、亮度、置信度、帧ID） |
| `trajectories` | 流星轨迹（多帧关联、速度、线性度） |
| `classifications` | 分类结果（流星/卫星/飞机/噪点、置信度） |

### 存储规则

- 上传图片 → `storage/uploads/{task_id}/`
- 检测结果 → `storage/results/{task_id}/`（标注图 + CSV + JSON）
- 日志 → `storage/logs/`，按天轮转
- 数据库 → `storage/meteo.db`

---

## 6. Web 界面

### 6.1 页面结构

| 页面 | 功能 |
|------|------|
| **Dashboard** | 任务列表、状态概览、最近检测结果缩略图 |
| **Upload** | 拖拽上传图片/目录，选择检测参数 |
| **任务详情** | 处理日志实时刷新、检测结果图、轨迹可视化、CSV 下载 |
| **历史记录** | 所有任务的检测结果浏览、搜索、对比 |
| **实时模式** | 启动/停止监控、实时检测画面、流星事件弹窗 |

### 6.2 技术细节

- 后端渲染：FastAPI + Jinja2 模板
- 前端交互：HTMX 局部刷新，无复杂 JS 构建
- 实时进度：Server-Sent Events (SSE)
- 图表：Chart.js（亮度曲线、轨迹分布图）

---

## 7. 任务队列

### 7.1 任务类型

| 任务 | 说明 | 耗时预估 |
|------|------|---------|
| `process_single` | 单张图片检测 | 秒级 |
| `process_batch` | 批量目录处理 | 分钟~小时级 |
| `start_realtime` | 启动实时监控 | 持续运行 |
| `stop_realtime` | 停止实时监控 | 秒级 |

### 7.2 状态流转

```
PENDING → RUNNING → [SUCCESS | FAILED | CANCELLED]
```

### 7.3 关键机制

- 长任务通过 SSE 推送进度到前端
- 实时模式用专用 Worker 进程，不阻塞其他任务
- 任务取消：Celery revoke + 检测引擎优雅退出（检查点机制）

---

## 8. 部署

### 8.1 Docker 支持

```
Dockerfile              # 基础镜像（Python 3.12 + 系统依赖）
docker-compose.yml      # FastAPI + Celery Worker + Redis
.dockerignore
```

### 8.2 部署方案

- 本地开发：`docker compose up` 一键启动全套
- Linux 服务器：同样的 compose 文件直接跑
- 扩展 Worker：`docker compose up --scale worker=N`
- Web 访问：本地 `localhost:8000`，服务器用 IP 或绑定域名
- 不需要域名也能用，后续有需求再绑

---

## 9. 现有 Bug 修复（融入重构）

| Bug | 修复方案 |
|-----|---------|
| `config.py` 使用 `np.pi` 但未 `import numpy` | 配置迁移到 YAML，运行时验证 |
| `validate_meteor_line` 用 ROI 对角线斜率作为预期值 | 改为基于亮度+形状+运动特征的评分制 |
| `main.py` 是调试脚本混在根目录 | 移到 `cli/` 或清理 |
| `detect_lines` 返回值不一致（有时 None 有时 lines） | 统一返回结构体 |
| 缺少亮度验证 | 加入检测流水线 |
| 无日志系统 | 用 `logging` 模块替换所有 `print()` |

---

## 10. 实施计划

### Phase 1 — 基础重构（1-2 周）
- 修复所有现有 Bug
- 搭建新目录结构
- 提取 `core/` 引擎（流水线、检测、跟踪）
- 配置迁移到 YAML
- 增加 logging

### Phase 2 — Web + 任务队列（1-2 周）
- FastAPI + Celery + Redis 搭建
- 基础 Web 界面（Dashboard + Upload + 任务详情）
- 批量处理任务
- Docker Compose 配置

### Phase 3 — 检测能力提升（1-2 周）
- 背景建模增强（MOG2/中值帧）
- 亮度验证 + 线段合并
- EXIF 自动读取
- RAW 格式支持

### Phase 4 — 分类 + 高级功能（1-2 周）
- 流星/卫星/飞机分类
- 卫星交叉验证
- 实时检测模式
- 历史记录搜索

---

## 11. 不做的（YAGNI）

- SPA 前端框架（React/Vue）— SSR + HTMX 够用
- 多用户/权限系统 — 单人使用不需要
- PostgreSQL — SQLite 够用，未来按需迁移
- Kubernetes — 单服务器 + Docker Compose 够用
- 域名 — IP 访问即可，有需求再绑
