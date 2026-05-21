"""비용 계산 유틸 (OpenAI Judge 호출)."""
from eval.config import OPENAI_PRICING


def calc_openai_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """USD 비용 계산."""
    pricing = OPENAI_PRICING.get(model)
    if not pricing:
        # 알 수 없는 모델은 0 (방어적)
        return 0.0
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]
