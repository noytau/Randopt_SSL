"""
Central registry for models, tasks, and methods.
To add a new model/task/method, just decorate the class with @register_*.
"""

from typing import Dict, Type, Any

# ── Registries ──────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, Type] = {}
TASK_REGISTRY: Dict[str, Type] = {}
METHOD_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """Decorator: @register_model("data2vec_audio")"""
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def register_task(name: str):
    """Decorator: @register_task("asr")"""
    def wrapper(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return wrapper


def register_method(name: str):
    """Decorator: @register_method("randopt")"""
    def wrapper(cls):
        METHOD_REGISTRY[name] = cls
        return cls
    return wrapper


# ── Discovery ───────────────────────────────────────────────────────────────

def list_available():
    """Print everything registered — useful for CLI help."""
    print("Models :", list(MODEL_REGISTRY.keys()))
    print("Tasks  :", list(TASK_REGISTRY.keys()))
    print("Methods:", list(METHOD_REGISTRY.keys()))


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def get_task(name: str, **kwargs):
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[name](**kwargs)


def get_method(name: str, **kwargs):
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}")
    return METHOD_REGISTRY[name](**kwargs)
