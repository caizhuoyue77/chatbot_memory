# 使用官方 Python 作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装 gcc 和其他必需的编译工具
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libz-dev \
    make \
    && rm -rf /var/lib/apt/lists/*  # 安装编译工具并清理缓存

# 复制本地项目文件到容器中
COPY . /app

# 安装项目依赖
# 先复制 requirements-backend.txt 文件，再安装依赖
COPY requirements-backend.txt /app/
RUN pip install --no-cache-dir -r requirements-backend.txt

# 安装新增的依赖
RUN pip install fastapi-cache redis slowapi numpy loguru

# 暴露端口（假设 FastAPI 默认运行在 8000 端口）
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "chat:app", "--host", "0.0.0.0", "--port", "8000"]

# uvicorn chat:app --host 0.0.0.0 --port 8000