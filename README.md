# Freedom Speech for Pipecat

Два сервиса для [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat): потоковое распознавание и потоковый синтез казахской, русской и английской речи через WebSocket API Freedom Speech (https://freedomspeech.kz/docs).

- `FreedomSpeechSTTService` (`WebsocketSTTService`): `WS /ws/asr`, вход PCM16 16 кГц моно (другая частота пайплайна пересэмплируется), `partial` -> `InterimTranscriptionFrame`, `final` -> `TranscriptionFrame` с языком от сервера. `VADUserStoppedSpeakingFrame` отправляет `{"type": "stop"}`, чтобы сервер закрыл фразу. Тишина-keepalive каждые 8 с без аудио, потому что сервер закрывает сокет после 15 с без аудио. Настройка `prettify` (форматирование сущностей, чисел, дат) меняет query сокета и ведёт к переподключению.
- `FreedomSpeechTTSService` (`WebsocketTTSService`): `WS /ws/tts`, модель `freedomspeech-flash` по умолчанию, `audio_chunk` (base64 PCM16 24 кГц) -> `TTSAudioRawFrame`, `audio_end` закрывает контекст. `requestId` запроса равен `context_id` pipecat, контекст на каждый `run_tts` (`reuse_context_id_within_turn=False`), прерывание шлёт `{"type": "cancel"}`. Настройки: `voice`, `model`, `language`, `emotion` (у flash эмоции и теги пока не подключены).

## Структура

```
pipecat/services/freedomspeech/__init__.py
pipecat/services/freedomspeech/stt.py
pipecat/services/freedomspeech/tts.py
tests/test_freedomspeech_stt.py
tests/test_freedomspeech_tts.py
examples/freedomspeech_bot.py
```

Раскладка повторяет репозиторий pipecat: файлы `pipecat/services/freedomspeech/*` кладутся в `src/pipecat/services/freedomspeech/`, тесты в `tests/`. Внешних зависимостей сверх базового `pipecat-ai` нет (только `websockets`, он уже в базе), поэтому extra в `pyproject.toml` не нужен.

## Использование

```python
from pipecat.services.freedomspeech import FreedomSpeechSTTService, FreedomSpeechTTSService
from pipecat.transcriptions.language import Language

stt = FreedomSpeechSTTService(api_key=os.getenv("FS_API_KEY"))
tts = FreedomSpeechTTSService(
    api_key=os.getenv("FS_API_KEY"),
    settings=FreedomSpeechTTSService.Settings(voice="tomiris", language=Language.KK),
)
```

Ключ только из переменной окружения; в код и логи не попадает.

## Тесты

```bash
pip install pipecat-ai pytest pytest-asyncio
ln -s $(pwd)/pipecat/services/freedomspeech <site-packages>/pipecat/services/freedomspeech
python -m pytest tests
```

Проверено на pipecat-ai 1.12.0, Python 3.13: 26 тестов на поддельном сокете, плюс одна живая проверка каждого класса против API (18.10.2026).

## Отправка в pipecat

1. Форк pipecat-ai/pipecat, ветка `feat/freedomspeech-services`.
2. Скопировать `pipecat/services/freedomspeech/` в `src/pipecat/services/freedomspeech/`, тесты в `tests/`.
3. Добавить сервисы в документацию (`docs/`), запись в `CHANGELOG.md`, пример в `examples/foundational/`.
4. PR по шаблону репозитория; CLA не требуется (BSD 2-Clause).
