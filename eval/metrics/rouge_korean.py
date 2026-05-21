"""ROUGE 점수 계산 (한국어, 형태소 기반)."""
from rouge_score import rouge_scorer
from konlpy.tag import Mecab

try:
    _mecab = Mecab()
except Exception:
    _mecab = None


def _tokenize_ko(text: str) -> str:
    """한국어 문장을 형태소 분리해서 공백으로 join.

    Mecab 없으면 문자 단위 fallback.
    """
    if _mecab is None:
        # fallback: 문자 단위
        return " ".join(text.replace(" ", ""))
    morphs = _mecab.morphs(text)
    return " ".join(morphs)


def compute_rouge(prediction: str, references: list[str]) -> dict:
    """ROUGE-1/2/L F1 점수 계산.

    multi-reference: 모든 reference 중 최대값 사용.
    """
    if not prediction or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=KoreanTokenizer(),
    )

    best = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref in references:
        scores = scorer.score(ref, prediction)
        for key in best:
            best[key] = max(best[key], scores[key].fmeasure)
    return best


class KoreanTokenizer:
    """rouge_scorer가 받는 tokenizer 인터페이스."""

    def tokenize(self, text: str) -> list[str]:
        if _mecab is not None:
            return _mecab.morphs(text)
        # fallback: 문자 단위
        return list(text.replace(" ", ""))
