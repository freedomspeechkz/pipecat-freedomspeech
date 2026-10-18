#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Freedom Speech services: Kazakh, Russian and English STT and TTS."""

from pipecat.services.freedomspeech.stt import FreedomSpeechSTTService
from pipecat.services.freedomspeech.tts import FreedomSpeechTTSService

__all__ = ["FreedomSpeechSTTService", "FreedomSpeechTTSService"]
