"""한국어 BERTScore 계산."""
from typing import Iterable
from bert_score import score as bert_score


# 한국어 BERT 모델 (고정)
DEFAULT_MODEL_TYPE = "klue/roberta-large"


def compute_bertscore(
    predictions: list[str],
    references: list[str | list[str]],
    model_type: str = DEFAULT_MODEL_TYPE,
    batch_size: int = 16,
) -> dict:
    """배치 BERTScore F1 계산.

    references는 str 또는 list[str] (multi-reference).
    multi-reference의 경우 가장 높은 점수 사용.
    """
    if not predictions:
        return {"f1": [], "precision": [], "recall": []}

    # multi-reference 처리: 가장 긴 reference list 길이만큼 expand
    if any(isinstance(r, list) for r in references):
        # 각 prediction에 대해 reference list 모두와 비교 후 max
        all_f1, all_p, all_r = [], [], []
        for pred, refs in zip(predictions, references):
            refs_list = refs if isinstance(refs, list) else [refs]
            P, R, F = bert_score(
                [pred] * len(refs_list),
                refs_list,
                model_type=model_type,
                lang="ko",
                batch_size=batch_size,
                verbose=False,
            )
            all_f1.append(float(F.max().item()))
            all_p.append(float(P.max().item()))
            all_r.append(float(R.max().item()))
        return {"f1": all_f1, "precision": all_p, "recall": all_r}

    P, R, F = bert_score(
        predictions,
        references,
        model_type=model_type,
        lang="ko",
        batch_size=batch_size,
        verbose=False,
    )
    return {
        "f1": [float(x) for x in F],
        "precision": [float(x) for x in P],
        "recall": [float(x) for x in R],
    }


def compute_bertscore_single(prediction: str, references: list[str]) -> float:
    """단일 prediction의 BERTScore F1 (multi-ref 최대값)."""
    result = compute_bertscore([prediction], [references])
    return result["f1"][0] if result["f1"] else 0.0
