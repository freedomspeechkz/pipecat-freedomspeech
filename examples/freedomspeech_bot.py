#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Echo bot on Freedom Speech STT and TTS.

What the user says in Kazakh, Russian or English is transcribed and spoken back.
Set FS_API_KEY and run with the development runner (needs `pipecat-ai[webrtc]`):

    python freedomspeech_bot.py
"""

import os

from loguru import logger

from pipecat.frames.frames import TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.freedomspeech import FreedomSpeechSTTService, FreedomSpeechTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


class Echo(FrameProcessor):
    """Turn every final transcript into a TTS request."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(TTSSpeakFrame(frame.text), FrameDirection.DOWNSTREAM)
        else:
            await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    api_key = os.environ["FS_API_KEY"]
    stt = FreedomSpeechSTTService(api_key=api_key)
    tts = FreedomSpeechTTSService(
        api_key=api_key,
        settings=FreedomSpeechTTSService.Settings(voice="tomiris", language=Language.KK),
    )
    pipeline = Pipeline([transport.input(), stt, Echo(), tts, transport.output()])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=runner_args.handle_sigint).run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point discovered by the pipecat development runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
