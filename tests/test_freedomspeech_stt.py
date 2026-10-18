#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.freedomspeech.stt import (
    FREEDOMSPEECH_ASR_SAMPLE_RATE,
    FreedomSpeechSTTService,
    language_to_freedomspeech_language,
)
from pipecat.transcriptions.language import Language


class _FakeWebSocket:
    """Minimal websocket double: records sends, replays queued messages."""

    def __init__(self, messages=()):
        self.state = State.OPEN
        self.sent = []
        self._messages = list(messages)

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return self._messages.pop(0)

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _service(sample_rate: int = 16000, **kwargs) -> FreedomSpeechSTTService:
    service = FreedomSpeechSTTService(api_key="test-key", sample_rate=sample_rate, **kwargs)
    # setup() normally copies the pipeline rate; tests skip the pipeline
    service._sample_rate = sample_rate
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.emit_stt_usage_metrics = AsyncMock()
    service._call_event_handler = AsyncMock()
    service._user_id = "user-1"
    return service


def test_language_mapping():
    assert language_to_freedomspeech_language(Language.KK) == "kk"
    assert language_to_freedomspeech_language(Language.RU_RU) == "ru"
    assert language_to_freedomspeech_language(Language.EN_US) == "en"


def test_default_settings_and_url():
    service = _service()
    assert service._settings.model == "freedomspeech"
    assert service._settings.language is None
    assert service._settings.prettify is True
    assert service._ws_url() == "wss://api.freedomspeech.kz/ws/asr?apiKey=test-key"


def test_prettify_false_goes_to_query():
    service = _service(settings=FreedomSpeechSTTService.Settings(prettify=False))
    assert service._ws_url().endswith("/ws/asr?apiKey=test-key&prettify=false")


def test_base_url_trailing_slash_is_stripped():
    service = _service(base_url="ws://localhost:8000/")
    assert service._ws_url().startswith("ws://localhost:8000/ws/asr?")


@pytest.mark.asyncio
async def test_run_stt_sends_raw_pcm():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket = ws

    pcm = b"\x01\x00" * 1600
    frames = [f async for f in service.run_stt(pcm)]

    assert frames == [None]
    assert ws.sent == [pcm]


@pytest.mark.asyncio
async def test_run_stt_drops_odd_trailing_byte():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket = ws

    [f async for f in service.run_stt(b"\x01\x00\x02")]

    assert ws.sent == [b"\x01\x00"]


@pytest.mark.asyncio
async def test_run_stt_resamples_when_pipeline_rate_differs():
    service = _service(sample_rate=48000)
    ws = _FakeWebSocket()
    service._websocket = ws

    # 20 chunks of 20 ms at 48 kHz; the stream resampler keeps a short delay,
    # so the total is compared with a tolerance rather than chunk by chunk
    chunk_48k = b"\x10\x00" * 960
    for _ in range(20):
        [f async for f in service.run_stt(chunk_48k)]

    total = sum(len(c) for c in ws.sent)
    expected = 20 * 2 * FREEDOMSPEECH_ASR_SAMPLE_RATE * 20 // 1000  # 20 x 20 ms at 16 kHz
    assert 0.8 * expected <= total <= expected
    assert all(len(c) % 2 == 0 for c in ws.sent)


@pytest.mark.asyncio
async def test_vad_stop_sends_stop_message():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket = ws
    service._reconnecting = False

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert json.loads(ws.sent[-1]) == {"type": "stop"}


@pytest.mark.asyncio
async def test_partial_and_final_messages_become_frames():
    service = _service()
    service._websocket = _FakeWebSocket(
        [
            json.dumps({"type": "ready"}),
            json.dumps({"type": "partial", "text": "сәлем"}),
            json.dumps({"type": "final", "text": "сәлеметсіз бе", "language": "kk"}),
        ]
    )

    await service._receive_messages()

    frames = [call.args[0] for call in service.push_frame.call_args_list]
    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert frames[0].text == "сәлем"
    assert frames[0].user_id == "user-1"
    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].text == "сәлеметсіз бе"
    assert frames[1].language == Language.KK
    service.emit_stt_usage_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_transcripts_are_ignored():
    service = _service()
    service._websocket = _FakeWebSocket(
        [json.dumps({"type": "partial", "text": ""}), json.dumps({"type": "final", "text": ""})]
    )

    await service._receive_messages()

    service.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_error_message_is_reported():
    service = _service()
    service._websocket = _FakeWebSocket([json.dumps({"type": "error", "message": "bad key"})])

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "bad key" in service.push_error.call_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_connect_waits_for_ready_frame():
    service = _service()
    ws = _FakeWebSocket([json.dumps({"type": "ready"})])
    service._websocket_connect = AsyncMock(return_value=ws)

    await service._connect_websocket()

    assert service._websocket is ws
    url = service._websocket_connect.call_args.args[0]
    assert url == "wss://api.freedomspeech.kz/ws/asr?apiKey=test-key"
    service._call_event_handler.assert_awaited_with("on_connected")


@pytest.mark.asyncio
async def test_connect_reports_error_frame_from_server():
    service = _service()
    ws = _FakeWebSocket([json.dumps({"type": "error", "message": "unauthorized"})])
    service._websocket_connect = AsyncMock(return_value=ws)

    await service._connect_websocket()

    assert service._websocket is None
    service.push_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_prettify_reconnects():
    service = _service()
    service._websocket = object()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._update_settings(FreedomSpeechSTTService.Settings(prettify=False))

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
    assert service._settings.prettify is False


@pytest.mark.asyncio
async def test_update_language_does_not_reconnect():
    service = _service()
    service._websocket = object()
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._update_settings(FreedomSpeechSTTService.Settings(language=Language.RU))

    service._disconnect.assert_not_awaited()
    assert service._settings.language == "ru"
