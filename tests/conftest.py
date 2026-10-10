"""Las pruebas nunca leen la lista real de modelos de OpenRouter: se usa esta, con la forma de su API
(/api/v1/models) y la situación del 10-oct-2026 (R1 y gpt-oss-120b ya solo de pago)."""
import pytest

import ai_hub

SAMPLE_MODELS = {"data": [
    {"id": "deepseek/deepseek-v4-flash-0731:free", "created": 1753920000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "deepseek/deepseek-v4-flash:free", "created": 1745452800, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "deepseek/deepseek-r1-0528", "created": 1748390400, "pricing": {"prompt": "0.0000005", "completion": "0.00000215"}},
    {"id": "openai/gpt-oss-120b", "created": 1754400000, "pricing": {"prompt": "0.0000001", "completion": "0.0000005"}},
    {"id": "google/gemma-4-31b-it:free", "created": 1756000000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "openrouter/free", "created": 1760000000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "openrouter/auto", "created": 1700000000, "pricing": {"prompt": "-1", "completion": "-1"}},
]}


@pytest.fixture(autouse=True)
def openrouter_free_list(monkeypatch):
    """Lista de gratis simulada y memoria limpia en cada prueba. Devuelve las lecturas hechas."""
    reads = []

    def fake(get, timeout=8):
        reads.append(timeout)
        return SAMPLE_MODELS

    ai_hub._FREE.update(t=0.0, ids=None, bad=set())
    monkeypatch.setattr(ai_hub, "_models_json", fake)
    yield reads
    ai_hub._FREE.update(t=0.0, ids=None, bad=set())
