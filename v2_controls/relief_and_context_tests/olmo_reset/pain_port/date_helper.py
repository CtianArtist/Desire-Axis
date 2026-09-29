import contextlib
from types import SimpleNamespace
C = SimpleNamespace(BASE_RUN_DATE=(2026, 9, 22))
@contextlib.contextmanager
def pinned_date(ymd=C.BASE_RUN_DATE):
    """Make chat templates that stamp today's date render the neutral-passages run's run date instead.

    ``transformers`` exposes ``strftime_now`` to templates through ``datetime.now()`` in its
    chat-template utilities; Llama 3.1 and Mistral Small 3 put that date in the system turn, so
    the ids of turn 0 depend on the day the session was generated.
    """
    try:
        import transformers.utils.chat_template_utils as ctu
    except Exception:  # noqa: BLE001
        yield
        return
    real = getattr(ctu, "datetime", None)
    if real is None or not isinstance(real, type):
        yield
        return
    y, m, d = ymd

    class _Fixed(real):  # type: ignore[misc,valid-type]
        @classmethod
        def now(cls, tz=None):
            return cls(y, m, d, 12, 0, 0, tzinfo=tz)

    ctu.datetime = _Fixed
    try:
        yield
    finally:
        ctu.datetime = real
