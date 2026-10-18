#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Freedom Speech speech-to-text service implementation.

Streaming recognition of Kazakh, Russian and English over the Freedom Speech
WebSocket API (``/ws/asr``). The client sends raw 16 kHz mono PCM16 frames and
receives ``partial`` and ``final`` transcripts; API reference:
https://freedomspeech.kz/docs.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# The server accepts exactly this input format and does not resample.
FREEDOMSPEECH_ASR_SAMPLE_RATE = 16000

# The server closes a socket after 15 seconds without audio; silence keepalives
# are sent before that happens.
DEFAULT_KEEPALIVE_TIMEOUT_S = 8.0

_LANGUAGE_MAP = {
    Language.KK: "kk",
    Language.RU: "ru",
    Language.EN: "en",
}

_SERVICE_TO_LANGUAGE = {"kk": Language.KK, "ru": Language.RU, "en": Language.EN}


def language_to_freedomspeech_language(language: Language | str) -> str | None:
    """Convert a Language enum to the Freedom Speech language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        ``kk``, ``ru`` or ``en``, or the base code of an unmapped language.
    """
    return resolve_language(cast("Language", language), _LANGUAGE_MAP, use_base_code=True)


@dataclass
class FreedomSpeechSTTSettings(STTSettings):
    """Settings for FreedomSpeechSTTService.

    ``model`` and ``language`` are inherited. The service detects the language
    on the server, so ``language`` only labels the frames when it is set.

    Parameters:
        prettify: Whether the server formats named entities, numbers and dates
            in the transcript. ``False`` returns the raw recognizer output.
    """

    prettify: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class FreedomSpeechSTTService(WebsocketSTTService):
    """Freedom Speech streaming speech-to-text service.

    Sends raw 16 kHz mono PCM16 audio over ``/ws/asr`` and pushes
    :class:`InterimTranscriptionFrame` for ``partial`` messages and
    :class:`TranscriptionFrame` for ``final`` messages. The language of a
    final transcript comes from the server (``kk``, ``ru`` or ``en``).

    A :class:`VADUserStoppedSpeakingFrame` sends ``{"type": "stop"}`` so the
    server closes the current phrase and emits its final transcript.

    Example::

        stt = FreedomSpeechSTTService(api_key=os.getenv("FS_API_KEY"))
    """

    Settings = FreedomSpeechSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://freedomspeech.kz",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        keepalive_timeout: float | None = DEFAULT_KEEPALIVE_TIMEOUT_S,
        **kwargs,
    ):
        """Initialize the Freedom Speech STT service.

        Args:
            api_key: Freedom Speech API key.
            base_url: WebSocket base URL. Defaults to ``wss://freedomspeech.kz``.
            sample_rate: Pipeline audio sample rate in Hz. Audio is resampled to
                16 kHz before sending because the server accepts only that rate.
            settings: Runtime-updatable settings (``prettify``, ``language``).
            keepalive_timeout: Seconds without audio before a silence keepalive
                is sent. The server drops idle sockets after 15 seconds.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        default_settings = self.Settings(
            model="freedomspeech",
            language=None,
            prettify=True,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            keepalive_timeout=keepalive_timeout,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._websocket = None
        self._receive_task = None
        self._resampler = create_stream_resampler()

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
        query: dict[str, Any] = {"apiKey": self._api_key}
        if assert_given(self._settings.prettify) is False:
            query["prettify"] = "false"
        return f"{self._base_url}/ws/asr?{urlencode(query)}"

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect when the socket query changes."""
        changed = await super()._update_settings(delta)
        if "prettify" in changed and self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service.

        Args:
            frame: End frame to stop processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service.

        Args:
            frame: Cancel frame to abort processing.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and close the phrase when the user stops speaking.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_stop()

    async def _send_stop(self) -> bool:
        """Ask the server to finalize the current phrase.

        Returns:
            Whether the message was sent.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return False
        try:
            await self._websocket.send(json.dumps({"type": "stop"}))
        except Exception as e:
            logger.warning(f"{self}: failed to send stop: {e}")
            return False
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the server.

        Args:
            audio: Raw PCM16 mono audio at the pipeline sample rate.

        Yields:
            None (transcripts arrive through the receive task).
        """
        rate = self.sample_rate or FREEDOMSPEECH_ASR_SAMPLE_RATE
        if rate != FREEDOMSPEECH_ASR_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, rate, FREEDOMSPEECH_ASR_SAMPLE_RATE)
        if len(audio) % 2:
            audio = audio[:-1]
        if audio and self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language | None):
        """Record transcription event for tracing."""
        pass

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Freedom Speech STT")

            websocket = await self._websocket_connect(
                self._ws_url(),
                additional_headers={"User-Agent": "pipecat-freedomspeech"},
                max_size=None,
            )
            self._websocket = websocket

            # The server sends {"type": "ready"} before it accepts audio.
            ready = json.loads(await websocket.recv())
            if ready.get("type") == "error":
                raise Exception(f"received error {ready.get('message')}")
            if ready.get("type") != "ready":
                raise Exception(f"unexpected first message type {ready.get('type')}")

            await self._call_event_handler("on_connected")
            logger.debug("Connected to Freedom Speech STT")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Freedom Speech: {e}", exception=e)

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Freedom Speech STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _frame_language(self, code: str | None) -> Language | None:
        if code and code in _SERVICE_TO_LANGUAGE:
            return _SERVICE_TO_LANGUAGE[code]
        configured = assert_given(self._settings.language)
        if configured in _SERVICE_TO_LANGUAGE:
            return _SERVICE_TO_LANGUAGE[cast("str", configured)]
        return cast("Language | None", configured) if isinstance(configured, Language) else None

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
                continue

            type_ = msg.get("type", "")
            if type_ == "partial":
                await self._handle_partial(msg.get("text", ""))
            elif type_ == "final":
                await self._handle_final(msg.get("text", ""), msg.get("language"))
            elif type_ == "ready":
                continue
            elif type_ == "error":
                await self.push_error(error_msg=f"Error: {msg.get('message', msg)}")

    async def _handle_partial(self, text: str):
        if not text:
            return
        await self.push_frame(
            InterimTranscriptionFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._frame_language(None),
            )
        )

    async def _handle_final(self, text: str, language_code: str | None):
        if not text:
            return
        language = self._frame_language(language_code)
        logger.debug(f"Final transcription: [{text}]")
        await self.emit_stt_usage_metrics()
        await self.push_frame(TranscriptionFrame(text, self._user_id, time_now_iso8601(), language))
        await self._trace_transcription(text, is_final=True, language=language)
