# astrbot_plugin_tts_emotion

AstrBot 插件：对 AI 回复进行情绪分析（8D 向量），并调用 IndexTTS2（OpenAI 兼容 `/v1/audio/speech`）合成带情绪的语音（`Record`）。

## 功能

- 自动将 AI 回复转成语音并追加到消息链
- 可选：启用情绪检测（使用当前会话的 LLM provider，生成 8 维情绪向量）
- 会话级开关：`/tts_emo`（只影响当前会话，不影响后台全局启用状态）
- 支持将 IndexTTS2 的 `x_advanced` 参数从后台配置透传

## 安装

将本仓库作为 AstrBot 插件安装到 `data/plugins/astrbot_plugin_tts_emotion`，然后在 AstrBot 后台启用插件并配置参数。

依赖：`aiohttp`（见 `requirements.txt`）。

## 使用

### 会话级开关

- `/tts_emo`：切换当前会话启用/关闭
- `/tts_emo on|off|toggle|status`

说明：所有以 `/` 开头的指令消息不会触发 TTS（避免影响 `/help` 等其它插件命令）。

### 输出策略

- 默认会保留原文字回复并追加语音
- 关闭 `dual_output` 后将仅发送语音（会从消息链里移除 `Plain` 文本组件）

## 情绪向量是如何生成的？

当启用 `enable_emotion` 时，插件会调用当前会话正在使用的 LLM provider：

- 输入：当前用户消息（截断 300）+ 当前 AI 回复（截断 500）
- 输出：8 维 JSON 数组，顺序为 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`
- 解析失败时回退到中性默认向量 `DEFAULT_EMOTION_VECTOR`

## 配置说明

配置由 AstrBot 后台根据 `_conf_schema.json` 提供的 Schema 展示。

### 基础配置

- `api_base_url`：IndexTTS2 API 基础地址（默认 `http://localhost:8000`）
- `api_key`：可选，Bearer Token
- `voice_id`：必填，IndexTTS2 的 `SpeechRequest.voice`
- `enable_emotion`：是否启用情绪检测
- `emotion_alpha`：情绪强度（0~1）
- `dual_output`：是否同时发送文字（默认 true）

### IndexTTS2 x_advanced（透传）

以下配置会被组装进请求体的 `x_advanced` 字段发送给 IndexTTS2：

- `max_mel_tokens`（默认 1500）
- `do_sample`（默认 true）
- `num_beams`（默认 3）
- `length_penalty`（默认 0）
- `diffusion_steps`（默认 25）
- `inference_cfg_rate`（默认 0.7）
- `max_tokens_per_segment`（默认 120）
- `temperature`（默认 0.8）
- `top_p`（默认 0.8）
- `top_k`（默认 30）
- `repetition_penalty`（默认 10）
- `interval_silence`（默认 200）

## 已知限制

- 语音会写入临时文件（`/tmp/*.wav`），当前版本不自动清理。

