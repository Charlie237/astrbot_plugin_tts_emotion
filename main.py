"""
AstrBot TTS Plugin with Emotion Detection

使用AI检测文本情绪，调用IndexTTS2 API生成带情绪的语音。
支持对话上下文感知的情绪分析。
"""

import json
import re
import tempfile
import aiohttp
from typing import Optional, List
from astrbot import logger as astr_logger
from astrbot.api.star import Context, Star, register
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter
from astrbot.api.message_components import Record, Plain


# 默认的中性情绪向量
# 向量格式: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
DEFAULT_EMOTION_VECTOR = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

PLUGIN_ID = "astrbot_plugin_tts_emotion"

# 情绪检测的提示词 - 带上下文分析
EMOTION_DETECTION_PROMPT = """分析以下对话的情绪，根据对话上下文和当前回复，输出8个0到1之间的数值，表示当前回复应该用什么情绪表达。

8个维度依次是：
1. happy (快乐)
2. angry (愤怒)
3. sad (悲伤)
4. afraid (恐惧)
5. disgusted (厌恶)
6. melancholic (忧郁)
7. surprised (惊讶)
8. calm (平静)

一句话可能包含多种情绪，请根据情绪强度给出合理的数值。所有数值之和不需要等于1。

{context_section}

当前用户消息: {user_message}

当前AI回复: {ai_reply}

请根据对话上下文和当前回复的内容，判断AI应该用什么情绪语气说这段话。
只输出JSON数组，格式如：[0.8, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.3]

情绪向量:"""


@register(PLUGIN_ID, "AstrBot", "TTS插件，支持AI情绪检测和带情绪的语音合成", "1.0.0")
class TTSEmotionPlugin(Star):
    """TTS插件，使用AI检测情绪并合成语音"""
    
    def __init__(self, context: Context):
        super().__init__(context)
        self.context = context
        self.logger = getattr(context, "logger", None) or astr_logger
        self._session_enabled: dict[str, bool] = {}
        
    def _obj_to_dict(self, obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
            try:
                return obj.model_dump()
            except Exception:
                return None
        if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
            try:
                return obj.dict()
            except Exception:
                return None
        return None

    def _get_plugin_config(self) -> dict:
        """
        获取当前插件的配置字典。

        AstrBot 的 `Context.get_config()` 在不同版本/场景下可能返回：
        - 整体配置（包含 `plugin_settings`）
        - 或仅返回当前插件的配置字典
        """
        config = self.context.get_config()
        config_dict = self._obj_to_dict(config) or {}

        plugin_settings = None
        if isinstance(config_dict, dict):
            plugin_settings = config_dict.get("plugin_settings")
        if plugin_settings is None and config is not None:
            plugin_settings = getattr(config, "plugin_settings", None)

        plugin_settings_dict = self._obj_to_dict(plugin_settings)
        if isinstance(plugin_settings_dict, dict):
            # 优先取当前插件 ID 的配置，避免旧插件 ID 的空配置覆盖新配置。
            for plugin_key in (PLUGIN_ID, "tts_emotion"):
                plugin_cfg = plugin_settings_dict.get(plugin_key)
                plugin_cfg_dict = self._obj_to_dict(plugin_cfg)
                if isinstance(plugin_cfg_dict, dict):
                    return plugin_cfg_dict

        if isinstance(config_dict, dict):
            return config_dict
        return {}

    def _get_config(self, key: str, default=None):
        """获取插件配置"""
        config = self._get_plugin_config()
        return config.get(key, default)
    
    async def _get_conversation_history(self, event: AstrMessageEvent, max_turns: int = 5) -> str:
        """获取对话历史上下文"""
        try:
            uid = event.unified_msg_origin
            conv_mgr = self.context.conversation_manager
            
            if not conv_mgr:
                return ""
            
            # 获取当前对话ID
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
            if not curr_cid:
                return ""
            
            # 获取对话对象
            conversation = await conv_mgr.get_conversation(uid, curr_cid)
            if not conversation or not conversation.history:
                return ""
            
            # history 是 JSON 字符串，需要解析
            try:
                history = json.loads(conversation.history)
            except (json.JSONDecodeError, TypeError):
                return ""
            
            if not history or not isinstance(history, list):
                return ""
            
            # 限制历史轮数
            recent_history = history[-max_turns * 2:] if len(history) > max_turns * 2 else history
            
            # 格式化历史记录
            context_lines = []
            for msg in recent_history:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                if isinstance(content, str) and content:
                    role_name = "用户" if role == "user" else "AI"
                    context_lines.append(f"{role_name}: {content[:200]}")  # 限制长度
            
            if context_lines:
                return "对话历史:\n" + "\n".join(context_lines)
            return ""
            
        except Exception as e:
            self.logger.warning(f"Failed to get conversation history: {e}")
            return ""
    
    def _parse_emotion_vector(self, response: str) -> Optional[List[float]]:
        """从LLM响应中解析8维情绪向量"""
        try:
            # 移除可能的markdown代码块标记
            cleaned = response.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'^```\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            cleaned = cleaned.strip()
            
            # 尝试找到JSON数组
            match = re.search(r'\[[\d.,\s]+\]', cleaned)
            if match:
                vector = json.loads(match.group())
                
                # 验证向量格式
                if isinstance(vector, list) and len(vector) == 8:
                    vector = [max(0.0, min(1.0, float(v))) for v in vector]
                    return vector
            
            return None
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self.logger.warning(f"Failed to parse emotion vector: {e}")
            return None
    
    async def _detect_emotion_vector(self, user_message: str, ai_reply: str, 
                                      context_section: str, event: AstrMessageEvent) -> List[float]:
        """使用LLM检测文本情绪，考虑对话上下文，直接返回8维向量"""
        try:
            # 获取当前会话使用的 LLM provider
            provider = self.context.get_using_provider(umo=event.unified_msg_origin)
            
            if not provider:
                self.logger.warning("No LLM provider available, using default emotion vector")
                return DEFAULT_EMOTION_VECTOR.copy()
            
            prompt = EMOTION_DETECTION_PROMPT.format(
                context_section=context_section if context_section else "（无历史对话）",
                user_message=user_message[:300],  # 限制长度
                ai_reply=ai_reply[:500]
            )
            
            # 使用 provider.text_chat 调用大模型
            llm_resp = await provider.text_chat(
                prompt=prompt,
                system_prompt="你是一个情绪分析助手，只输出JSON数组格式的情绪向量。"
            )
            
            # 获取响应文本
            response_text = llm_resp.completion_text if hasattr(llm_resp, 'completion_text') else str(llm_resp)
            
            # 解析响应，提取情绪向量
            vector = self._parse_emotion_vector(response_text)
            
            if vector:
                return vector
            
            self.logger.warning(f"Could not parse emotion vector from response: {response_text[:100]}")
            return DEFAULT_EMOTION_VECTOR.copy()
            
        except Exception as e:
            self.logger.error(f"Emotion detection failed: {e}")
            return DEFAULT_EMOTION_VECTOR.copy()
    
    async def _synthesize_speech(self, text: str, emotion_vector: List[float]) -> Optional[bytes]:
        """调用IndexTTS2 API合成语音"""
        api_base_url = self._get_config("api_base_url", "http://localhost:8000")
        api_key = self._get_config("api_key", "")
        voice_id = (
            self._get_config("voice_id", "")
            or self._get_config("voice", "")
            or ""
        )
        voice_name = str(self._get_config("voice_name", "") or "").strip()
        response_format = str(self._get_config("response_format", "wav") or "wav").lower()
        if response_format != "wav":
            response_format = "wav"
        emotion_alpha = self._get_config("emotion_alpha", 0.65)
        
        headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async def resolve_voice_id(session: aiohttp.ClientSession) -> str:
            if voice_id:
                return str(voice_id).strip()
            try:
                url = f"{api_base_url.rstrip('/')}/v1/audio/voices"
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"List voices failed: {response.status} - {error_text}")
                        return ""
                    data = await response.json()
            except Exception as e:
                self.logger.error(f"List voices request failed: {e}")
                return ""

            voices = data.get("voices") if isinstance(data, dict) else None
            if not isinstance(voices, list) or not voices:
                self.logger.error("No voices returned from /v1/audio/voices; please configure voice_id")
                return ""

            if voice_name:
                for v in voices:
                    if isinstance(v, dict) and str(v.get("name", "")).strip() == voice_name:
                        return str(v.get("id", "")).strip()

            first = voices[0]
            if isinstance(first, dict) and first.get("id"):
                return str(first["id"]).strip()
            return ""
        
        url = f"{api_base_url.rstrip('/')}/v1/audio/speech"

        payload = {
            "model": "indextts2",
            "input": text,
            "voice": "",
            "response_format": response_format,
            "x_emotion": {
                "type": "vector",
                "vector": emotion_vector,
                "alpha": emotion_alpha
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                resolved_voice_id = await resolve_voice_id(session)
                if not resolved_voice_id:
                    self.logger.error(
                        "Voice ID not configured; set `voice_id` (or `voice`) in plugin config, "
                        "or ensure /v1/audio/voices returns at least one voice."
                    )
                    return None
                payload["voice"] = resolved_voice_id
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        error_text = await response.text()
                        self.logger.error(
                            f"TTS API error: {response.status} - {error_text}"
                        )
                        return None
        except Exception as e:
            self.logger.error(f"TTS API request failed: {e}")
            return None
    
    def _save_audio_to_temp(self, audio_data: bytes, format: str = "wav") -> str:
        """将音频数据保存到临时文件"""
        suffix = f".{format}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_data)
            return f.name
    
    def _extract_text_from_chain(self, chain: list) -> str:
        """从消息链中提取纯文本"""
        texts = []
        for component in chain:
            if isinstance(component, Plain):
                texts.append(component.text)
        return "".join(texts)

    async def _get_session_key(self, event: AstrMessageEvent) -> str:
        uid = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager
        if not conv_mgr:
            return uid
        try:
            curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        except Exception:
            curr_cid = None
        return f"{uid}:{curr_cid}" if curr_cid else uid

    async def _is_enabled_for_session(self, event: AstrMessageEvent) -> bool:
        key = await self._get_session_key(event)
        if key in self._session_enabled:
            return self._session_enabled[key]
        return True

    @filter.command("tts_emo")
    async def cmd_tts_emo(self, event: AstrMessageEvent):
        """
        /tts_emo [on|off|toggle|status]

        仅对当前会话生效（内存态），不影响后台的全局插件启用状态。
        """
        msg = (event.message_str or "").strip()
        parts = msg.split()
        action = parts[1].lower() if len(parts) >= 2 else "toggle"

        session_key = await self._get_session_key(event)
        current = self._session_enabled.get(session_key, True)

        if action in {"on", "enable", "1", "true", "开启"}:
            new_val = True
        elif action in {"off", "disable", "0", "false", "关闭"}:
            new_val = False
        elif action in {"status", "show", "?"}:
            new_val = current
        else:
            new_val = not current

        self._session_enabled[session_key] = new_val

        status_text = "已开启" if new_val else "已关闭"
        # 兼容不同 AstrBot 版本的结果 API
        if hasattr(event, "plain_result") and callable(getattr(event, "plain_result")):
            yield event.plain_result(f"TTS Emotion（当前会话）{status_text}")
        else:
            yield [Plain(f"TTS Emotion（当前会话）{status_text}")]
    
    @filter.on_decorating_result()
    async def convert_to_speech(self, event: AstrMessageEvent):
        """
        在LLM响应发送前，将文本转换为语音
        
        这个装饰器钩子在消息发送前被调用，用于装饰/修改消息内容。
        通过 event.get_result().chain 获取并修改消息链。
        """
        # 获取消息结果
        result = event.get_result()
        if not result or not result.chain:
            return
        
        # 从消息链中提取AI回复文本
        ai_reply = self._extract_text_from_chain(result.chain)
        if not ai_reply:
            return
        
        # 获取用户消息
        user_message = event.message_str or ""

        # 命令回执不做 TTS，避免产生干扰
        if re.match(r"^\s*/?tts_emo\b", user_message):
            return

        if not await self._is_enabled_for_session(event):
            return
        
        enable_emotion = self._get_config("enable_emotion", True)
        
        # 检测情绪向量
        if enable_emotion:
            self.logger.info(f"Detecting emotion for reply: {ai_reply[:50]}...")
            
            # 获取对话历史上下文
            context_section = await self._get_conversation_history(event)
            
            emotion_vector = await self._detect_emotion_vector(
                user_message=user_message,
                ai_reply=ai_reply,
                context_section=context_section,
                event=event
            )
            self.logger.info(f"Detected emotion vector: {emotion_vector}")
        else:
            emotion_vector = DEFAULT_EMOTION_VECTOR.copy()
        
        # 合成语音
        self.logger.info("Synthesizing speech...")
        audio_data = await self._synthesize_speech(ai_reply, emotion_vector)
        
        if not audio_data:
            self.logger.warning("Failed to synthesize speech, keeping text response")
            return
        
        # 保存音频到临时文件
        response_format = self._get_config("response_format", "wav")
        audio_path = self._save_audio_to_temp(audio_data, response_format)
        self.logger.info(f"Audio saved to: {audio_path}")
        
        # 创建音频消息组件并添加到消息链
        record = Record(file=audio_path)
        result.chain.append(record)
        
        self.logger.info("Audio message appended to response chain")
