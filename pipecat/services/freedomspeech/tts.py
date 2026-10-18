#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Freedom Speech text-to-speech service implementation.

Streaming synthesis of Kazakh, Russian and English over the Freedom Speech
WebSocket API (``/ws/tts``). Audio arrives as base64 PCM16 chunks at 24 kHz;
API reference: https://freedomspeech.kz/docs.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

FREEDOMSPEECH_TTS_SAMPLE_RATE = 24000

_LANGUAGE_MAP = {
    Language.KK: "kk",
    Language.RU: "ru",
    Language.EN: "en",
}


def language_to_freedomspeech_language(language: Language | str) -> str | None:
    """Convert a Language enum to the Freedom Speech language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        ``kk``, ``ru`` or ``en``, or the base code of an unmapped language.
    """
    return resolve_language(cast("Language", language), _LANGUAGE_MAP, use_base_code=True)


@dataclass
class FreedomSpeechTTSSettings(TTSSettings):
    """Settings for FreedomSpeechTTSService.

    ``model``, ``voice`` and ``language`` are inherited.

    Parameters:
        emotion: Emotion for voices that support it (``neutral``, ``happy``,
            ``sad``, ``surprised``, ``angry``, ``fearful``). The
            ``freedomspeech-flash`` model ignores it.
    """

    emotion: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class FreedomSpeechTTSService(WebsocketTTSService):
    """Freedom Speech WebSocket text-to-speech service.

    Each ``run_tts`` call sends one ``synthesize`` request whose ``requestId``
    is the pipecat context id, so audio chunks map back to their context. The
    server queues up to 8 requests per socket. An interruption sends
    ``cancel`` which drops the current request and the queue.

    Example::

        tts = FreedomSpeechTTSService(
            api_key=os.getenv("FS_API_KEY"),
            settings=FreedomSpeechTTSService.Settings(voice="tomiris", language=Language.KK),
        )
    """

    Settings = FreedomSpeechTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.freedomspeech.kz",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Freedom Speech TTS service.

        Args:
            api_key: Freedom Speech API key.
            base_url: WebSocket base URL. Defaults to ``wss://api.freedomspeech.kz``.
            sample_rate: Output sample rate. The server produces 24 kHz; other
                values are resampled by the pipeline.
            settings: Runtime-updatable settings (``voice``, ``model``,
                ``language``, ``emotion``).
            **kwargs: Additional arguments passed to the parent WebsocketTTSService.
        """
        default_settings = self.Settings(
            model="freedomspeech-flash",
            voice="tomiris",
            language=None,
            emotion=None,
        )
        if settings is not None:
            if isinstance(settings.language, Language):
                settings = settings.copy()
                settings.language = language_to_freedomspeech_language(settings.language)
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate or FREEDOMSPEECH_TTS_SAMPLE_RATE,
            pause_frame_processing=True,
            push_stop_frames=False,
            push_start_frame=True,
            reuse_context_id_within_turn=False,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._websocket = None
        self._receive_task = None
        self._server_sample_rate = FREEDOMSPEECH_TTS_SAMPLE_RATE

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as the service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to the Freedom Speech language code."""
        return language_to_freedomspeech_language(language)

    def _ws_url(self) -> str:
        return f"{self._base_url}/ws/tts?{urlencode({'apiKey': self._api_key})}"

    def _build_request(self, text: str, request_id: str) -> str:
        req: dict[str, Any] = {"type": "synthesize", "text": text, "requestId": request_id}
        voice = assert_given(self._settings.voice)
        model = assert_given(self._settings.model)
        language = assert_given(self._settings.language)
        emotion = assert_given(self._settings.emotion)
        if voice:
            req["voice"] = voice
        if model:
            req["model"] = model
        if language:
            req["language"] = language
        if emotion:
            req["emotion"] = emotion
        return json.dumps(req, ensure_ascii=False)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Freedom Speech TTS")
            self._websocket = await self._websocket_connect(
                self._ws_url(),
                additional_headers={"User-Agent": "pipecat-freedomspeech"},
                max_size=None,
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}", exception=e))
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Freedom Speech TTS")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error_frame(ErrorFrame(error=f"{self} error: {e}"))
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the current request and the server queue on interruption.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self.stop_all_metrics()
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception as e:
                logger.error(f"{self} error sending cancel: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON message: {message}")
                continue

            ctx_id = msg.get("requestId")
            msg_type = msg.get("type")
            if msg_type == "audio_chunk":
                if not ctx_id or not self.audio_context_available(ctx_id):
                    continue
                sr = msg.get("sampleRate")
                if isinstance(sr, int) and sr > 0:
                    self._server_sample_rate = sr
                audio = msg.get("audio")
                if not audio:
                    continue
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(audio),
                    sample_rate=self._server_sample_rate,
                    num_channels=1,
                    context_id=ctx_id,
                )
                await self.append_to_audio_context(ctx_id, frame)
            elif msg_type in ("audio_end", "cancelled"):
                if not ctx_id or not self.audio_context_available(ctx_id):
                    continue
                await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                await self.remove_audio_context(ctx_id)
            elif msg_type == "error":
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg.get('code', '')} {msg.get('message', msg)}")
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.push_frame(TTSStoppedFrame(context_id=ctx_id))
                    await self.remove_audio_context(ctx_id)
                self.reset_active_audio_context()
            else:
                logger.trace(f"{self}: ignoring message type {msg_type}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one synthesis request.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames; used as the
                server ``requestId``.

        Yields:
            Frame: None (audio frames are appended to the context by the
            receive task) or an ErrorFrame.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._get_websocket().send(self._build_request(text, context_id))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
