# EmoAgent Router Agent

当前版本的 `EmoAgent` 通过调用大模型deepseek-chat进行判断。(若需调用其他大模型，请修改各个agent的base_url及model或者设置环境变量)

## 情绪分析服务启动

项目中提供了一个基于 FastAPI 的情绪分析服务，入口文件是 [service/app.py](/d:/PracticalTraining/Agenttest/EmoAgent/service/app.py:1)。服务统一封装了以下几个 Agent：

- `Router Agent`
- `Emotion Agent`
- `Sarcasm Agent`
- `Mix Agent`
- `Judge Agent`

### 1. 启动前准备

服务启动时会优先读取项目根目录 `.env` 中的配置，也支持直接读取系统环境变量。至少需要准备：

```env
API_KEY=你的deepseek服务密钥
```

说明：

- `API_KEY` 必填
- `LLM_BASE_URL` 和 `LLM_MODEL` 不填时会使用 `service/app.py` 中的默认值
- 如果 `API_KEY` 缺失，服务会进入 `degraded` 状态，`/health` 会返回 `ready: false`

### 2. 创建并进入虚拟环境

仓库中没有提交 `.venv`。首次拉取项目后，请在项目根目录创建你自己的虚拟环境：

```powershell
cd service
python -m venv .venv
```

### 3. 激活虚拟环境

在 PowerShell 中执行：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 因执行策略无法激活，也可以不激活，直接使用虚拟环境里的 Python 或 `uvicorn`，见后文“备用启动方式”。

### 4. 安装依赖

激活虚拟环境后，在项目根目录安装依赖：

```powershell
pip install -r requirements.txt
```

### 5. 进入 service 目录并启动服务

激活虚拟环境后执行：

```powershell
uvicorn app:app --reload
```

启动成功后，默认访问地址为：

```text
http://127.0.0.1:8000
```

若要指定端口，则执行:

```powershell
uvicorn app:app --reload --port "指定端口”
```

### 6. 健康检查

服务启动后可以先访问健康检查接口：

```powershell
curl http://127.0.0.1:8000/health
```

正常情况下返回：

```json
{
  "status": "ok",
  "ready": true
}
```

如果配置缺失，可能返回：

```json
{
  "status": "degraded",
  "ready": false,
  "reason": "API_KEY not found. Please set it in .env or environment variables."
}
```

### 7. 可用接口

当前服务提供以下接口：

- `GET /health`
- `POST /router`
- `POST /emotion`
- `POST /sarcasm`
- `POST /mix`
- `POST /judge`

其中：

- `/router`、`/emotion`、`/sarcasm`、`/mix` 接收文本输入
- `/judge` 接收各上游 Agent 的结构化结果并输出最终裁决

### 8. 文本类接口请求示例

适用于 `/router`、`/emotion`、`/sarcasm`、`/mix`：

```json
{
  "id": "msg_001",
  "user_id": "u_1001",
  "text": "太好了，周末又能继续改需求了。",
  "source": "chat",
  "created_at": "2026-03-24T14:00:00",
  "metadata": {}
}
```

PowerShell 调用示例：

```powershell
curl -Method Post `
  -Uri http://127.0.0.1:8000/router `
  -ContentType "application/json" `
  -Body '{
    "id":"msg_001",
    "user_id":"u_1001",
    "text":"太好了，周末又能继续改需求了。",
    "source":"chat",
    "created_at":"2026-03-24T14:00:00",
    "metadata":{}
  }'
```

### 9. Judge 接口请求示例

`/judge` 需要传入上游 Agent 的结构化结果，即在Body里放入例如：

```json
{
  "text": "太好了，周末又能继续改需求了。",
  "router_result": {
    "sample_type": "sarcasm_suspected",
    "need_sarcasm_check": true,
    "need_mix_check": false,
    "routing_reason": "句子表面正向，但事件语境明显负向，疑似反讽。",
    "evidence": ["正向词: 太好了", "负向场景: 周末继续改需求"]
  },
  "emotion_result": {
    "emotion": "开心",
    "intensity": 62,
    "confidence": 0.72,
    "reason": "文本表面包含明显正向表达。"
  },
  "sarcasm_result": {
    "is_sarcasm": true,
    "surface_emotion": "开心",
    "true_emotion": "厌烦",
    "revised_intensity": 74,
    "confidence": 0.86,
    "reason": "正向词与负向工作场景形成反差。"
  },
  "mix_result": null
}
```

### 10. 备用启动方式

如果你不想激活虚拟环境，可以直接运行虚拟环境中的 `uvicorn`：

```powershell
cd d:\PracticalTraining\Agenttest\EmoAgent\service
..\.venv\Scripts\uvicorn.exe app:app --reload
```

也可以直接使用虚拟环境里的 Python：

```powershell
cd d:\PracticalTraining\Agenttest\EmoAgent\service
..\.venv\Scripts\python.exe -m uvicorn app:app --reload
```
