# ☄️ 流星探测 (Meteor Detection)

从夜空中自动检测流星轨迹的 Python Web 应用。支持单张/批量/实时三种检测模式，具备完整的前端管理界面和任务队列。

## 功能特性

- **多种检测模式**：单张检测、批量处理、实时文件夹监听
- **多种背景差分**：简单差分、图像配准、MOG2、中值帧背景
- **线段合并**：自动合并共线/近似线段
- **轨迹跟踪**：多帧轨迹关联验证
- **智能分类**：基于运动特征区分流星/卫星/飞机/噪点
- **EXIF 读取**：自动提取拍摄时间、GPS、相机参数
- **Web 界面**：任务看板、结果浏览、历史搜索
- **任务队列**：Celery + Redis 异步处理
- **Docker 支持**：一键部署

## 快速开始

### 方式一：Docker Compose（推荐）

```bash
# 启动全套服务（Web + Worker + Redis）
docker compose up -d

# 访问 Web 界面
open http://localhost:8000
```

### 方式二：本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 或使用 poetry
poetry install

# 2. 启动 Redis
brew install redis
brew services start redis

# 3. 启动 Celery Worker（终端 1）
celery -A worker.celery_app worker --loglevel=info

# 4. 启动 Web 服务（终端 2）
uvicorn api.main:app --reload --port 8000

# 5. 访问
open http://localhost:8000
```

### 方式三：CLI

```bash
# 单张检测
python -m cli.meteo detect-single path/to/image.jpg --background path/to/bg.jpg

# 批量检测
python -m cli.meteo detect-batch path/to/images/ --background path/to/bg.jpg

# 鱼眼校正
python -m cli.meteo rectify input.jpg output.jpg

# 图像叠加
python -m cli.meteo merge path/to/images/ output.jpg

# 下载卫星 TLE 数据（供卫星识别交叉验证 / SatelliteFilter 使用）
python -m cli.meteo fetch-tle --group visual --group stations
# 仅刷新缺失或超过 7 天的 TLE
python -m cli.meteo fetch-tle --if-older 7
```

## API 使用

```bash
# 创建检测任务
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "batch",
    "params": {
      "image_dir": "/path/to/images",
      "background_path": "/path/to/background.jpg"
    }
  }'

# 查看任务列表
curl http://localhost:8000/api/tasks/

# 查看任务详情
curl http://localhost:8000/api/tasks/{task_id}

# 搜索检测结果
curl "http://localhost:8000/api/tasks/search/detections?object_class=meteor&min_confidence=0.5"

# 上传图片
curl -X POST http://localhost:8000/api/upload/ \
  -F "file=@/path/to/image.jpg"
```

## 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/test_detection/ -v
python -m pytest tests/test_classification/ -v
```

## 项目结构

```
meteo-detection/
├── api/                    # FastAPI Web 层
│   ├── main.py             # 应用入口
│   ├── routes/             # API 路由（tasks, upload）
│   ├── models/             # 数据模型 + 数据库
│   └── templates/          # Jinja2 模板
├── worker/                 # Celery Worker 层
│   ├── celery_app.py       # Celery 配置
│   ├── tasks.py            # 任务定义
│   └── runners/            # 运行器（realtime）
├── core/                   # 核心检测引擎
│   ├── pipeline.py         # 流水线编排
│   ├── preprocessing/      # 预处理（鱼眼校正、背景差分、叠加）
│   ├── detection/          # 检测（直线检测、轨迹跟踪、AI 模型）
│   ├── classification/     # 分类（流星/卫星/飞机）
│   └── utils/              # 工具（配置、日志、EXIF、可视化）
├── cli/                    # CLI 入口
│   └── meteo.py            # Click 命令行
├── config.yaml             # 统一配置
├── docker-compose.yml      # Docker 编排
├── tests/                  # 测试
└── storage/                # 运行时数据（上传、结果、日志）
```

## 配置

所有配置在 `config.yaml` 中管理：

```yaml
paths:
  base_dir: .
  storage_dir: storage

hough_line:
  threshold: 30
  min_line_length: 30

meteor_detection:
  score_threshold: 0.5
  score_weights:
    length: 0.3
    brightness: 0.4
    aspect_ratio: 0.3

tracking:
  max_frames: 5
  min_trajectory_length: 3

redis:
  host: localhost
  port: 6379

web:
  host: 0.0.0.0
  port: 8000
```

## 技术栈

| 层 | 技术 |
|----|------|
| Web | FastAPI + Jinja2 + HTMX |
| 任务队列 | Celery + Redis |
| 数据库 | SQLite |
| 检测引擎 | OpenCV + NumPy + astroalign |
| AI 模型 | TensorFlow (SSD MobileNet) |
| 部署 | Docker Compose |
| CLI | Click |
| 测试 | pytest |

## 许可证

MIT
