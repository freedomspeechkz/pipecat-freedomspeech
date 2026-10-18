#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.freedomspeech.tts import (
    FREEDOMSPEECH_TTS_SAMPLE_RATE,
    FreedomSpeechTTSService,
    language_to_freedomspeech_language,
)
from pipecat.transcriptions.language import Language


class _FakeWebSocket:
    def __init__(self, messages=()):
        self.state = State.OPEN
        self.sent = []
        self._messages = list(messages)

    async def send(self, data):
        self.sent.append(data)

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._messages:
            raise StopAsyncIteration
        return self._messages.pop(0)


def _service(**kwargs) -> FreedomSpeechTTSService:
    service = FreedomSpeechTTSService(api_key="test-key", **kwargs)
    service.push_frame = AsyncMock()
    service.push_error = AsyncMock()
    service.push_error_frame = AsyncMock()
    service.start_tts_usage_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    service.stop_all_metrics = AsyncMock()
    service.append_to_audio_context = AsyncMock()
    service.remove_audio_context = AsyncMock()
    service._call_event_handler = AsyncMock()
    return service


def test_language_mapping():
    assert language_to_freedomspeech_language(Language.KK_KZ) == "kk"
    assert language_to_freedomspeech_language(Language.RU) == "ru"
    assert language_to_freedomspeech_language(Language.EN) == "en"


def test_defaults():
    service = _service()
    assert service._settings.model == "freedomspeech-flash"
    assert service._settings.voice == "tomiris"
    assert service._settings.language is None
    assert service._settings.emotion is None
    # setup() copies this into sample_rate; before setup only the init value is known
    assert service._init_sample_rate == FREEDOMSPEECH_TTS_SAMPLE_RATE
    assert service._ws_url() == "wss://freedomspeech.kz/ws/tts?apiKey=test-key"


def test_settings_language_enum_is_converted():
    service = _service(
        settings=FreedomSpeechTTSService.Settings(
            voice="taymas", model="freedomspeech", language=Language.KK, emotion="happy"
        )
    )
    req = json.loads(service._build_request("Сәлем", "ctx-1"))
    assert req == {
        "type": "synthesize",
        "text": "Сәлем",
        "requestId": "ctx-1",
        "voice": "taymas",
        "model": "freedomspeech",
        "language": "kk",
        "emotion": "happy",
    }


def test_request_omits_unset_fields():
    req = json.loads(_service()._build_request("Привет", "ctx-2"))
    assert req == {
        "type": "synthesize",
        "text": "Привет",
        "requestId": "ctx-2",
        "voice": "tomiris",
        "model": "freedomspeech-flash",
    }


@pytest.mark.asyncio
async def test_run_tts_sends_request_and_yields_none():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket = ws

    frames = [f async for f in service.run_tts("Сәлем", "ctx-3")]

    assert frames == [None]
    assert json.loads(ws.sent[0])["requestId"] == "ctx-3"
    service.start_tts_usage_metrics.assert_awaited_once_with("Сәлем")


@pytest.mark.asyncio
async def test_run_tts_send_failure_yields_error_and_reconnects():
    service = _service()
    ws = _FakeWebSocket()
    ws.send = AsyncMock(side_effect=RuntimeError("boom"))
    service._websocket = ws
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    frames = [f async for f in service.run_tts("x", "ctx-4")]

    assert isinstance(frames[0], ErrorFrame)
    assert isinstance(frames[1], TTSStoppedFrame)
    assert frames[1].context_id == "ctx-4"
    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_chunk_and_end_map_to_context():
    service = _service()
    service.audio_context_available = lambda ctx: ctx == "ctx-5"
    pcm = b"\x01\x00\x02\x00"
    service._websocket = _FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "audio_chunk",
                    "requestId": "ctx-5",
                    "audio": base64.b64encode(pcm).decode(),
                    "chunkIndex": 0,
                    "sampleRate": 24000,
                    "format": "pcm16",
                }
            ),
            json.dumps({"type": "audio_chunk", "requestId": "other", "audio": "AAAA"}),
            json.dumps({"type": "audio_end", "requestId": "ctx-5", "totalChunks": 1, "sampleRate": 24000}),
        ]
    )

    await service._receive_messages()

    calls = service.append_to_audio_context.call_args_list
    assert len(calls) == 2
    ctx, frame = calls[0].args
    assert ctx == "ctx-5"
    assert isinstance(frame, TTSAudioRawFrame)
    assert frame.audio == pcm
    assert frame.sample_rate == 24000
    assert frame.num_channels == 1
    assert frame.context_id == "ctx-5"
    ctx, frame = calls[1].args
    assert isinstance(frame, TTSStoppedFrame)
    service.remove_audio_context.assert_awaited_once_with("ctx-5")
    service.stop_ttfb_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_closes_context():
    service = _service()
    service.audio_context_available = lambda ctx: True
    service._websocket = _FakeWebSocket([json.dumps({"type": "cancelled", "requestId": "ctx-6", "totalChunks": 2})])

    await service._receive_messages()

    ctx, frame = service.append_to_audio_context.call_args.args
    assert ctx == "ctx-6" and isinstance(frame, TTSStoppedFrame)
    service.remove_audio_context.assert_awaited_once_with("ctx-6")


@pytest.mark.asyncio
async def test_error_message_is_reported():
    service = _service()
    service.audio_context_available = lambda ctx: False
    service._websocket = _FakeWebSocket(
        [json.dumps({"type": "error", "code": "queue_full", "message": "too many requests"})]
    )

    await service._receive_messages()

    service.push_error.assert_awaited_once()
    assert "queue_full" in service.push_error.call_args.kwargs["error_msg"]


@pytest.mark.asyncio
async def test_interruption_sends_cancel():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket = ws
    service.remove_audio_context = AsyncMock()

    await service.on_audio_context_interrupted("ctx-7")

    assert json.loads(ws.sent[0]) == {"type": "cancel"}


@pytest.mark.asyncio
async def test_connect_uses_api_key_query():
    service = _service()
    ws = _FakeWebSocket()
    service._websocket_connect = AsyncMock(return_value=ws)

    await service._connect_websocket()

    assert service._websocket is ws
    assert service._websocket_connect.call_args.args[0] == "wss://freedomspeech.kz/ws/tts?apiKey=test-key"
    service._call_event_handler.assert_awaited_with("on_connected")
