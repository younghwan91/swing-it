"""6단계 위험모델을 북에서 쓰는 도우미 — 부분 모델과 능동위험. 모델 빌드는 :mod:`swing_it.book.riskmodel`, 값은 core ``build_risk_model`` 이 만든다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAR_TOL = 1e-12  # 이 아래로 음수면 반올림이 아니다


@dataclass
class Sub:
    """위험모델 부분집합(행렬은 지평 단위). core ``optimize`` 가 받는 모양과 같다."""

    X: np.ndarray
    F: np.ndarray
    D: np.ndarray
    names: list


def sub_model(model, rows: np.ndarray, h: int) -> Sub:
    """모델 ``(idx, X, F, D, names)`` 에서 행 ``rows``(전체 종목 번호)만, 지평 h 로 늘린 부분 모델."""
    if h <= 0:
        raise ValueError(f"지평 h 는 양수여야 한다: {h}")
    idx, X, F, D, names = model
    rows = np.asarray(rows)
    for what, arr in (("모델 좌표", np.asarray(idx)), ("rows", rows)):
        if arr.size and arr.dtype.kind not in "iu":     # int("005930") 은 조용히 5930 이 되어 다른 행을 집는다
            raise ValueError(f"{what}는 전체 패널의 행 번호(정수)여야 한다 — 종목 코드가 아니다: {arr.dtype}")
    pos = {int(c): i for i, c in enumerate(idx)}
    p = np.array([pos[int(r)] for r in rows], dtype=int)
    return Sub(X=X[p], F=h * F, D=h * D[p], names=list(names))


def active_risk(sub: Sub, d: np.ndarray) -> float:
    y = sub.X.T @ d
    var = float(y @ sub.F @ y + (d * d) @ sub.D)
    if var < -VAR_TOL:
        raise ValueError(f"음의 분산 {var:.3e} — F 가 양반정치가 아니다. 0 으로 자르면 공짜 위험이 된다")
    return float(np.sqrt(max(var, 0.0)))


def factor_share(sub: Sub, d: np.ndarray) -> float:
    """사전 능동분산 중 팩터 몫 yᵀFy / (yᵀFy + Σ d_i² D_i) (PO-3.4 — 지평 배수와 무관)."""
    y = sub.X.T @ d
    fv = float(y @ sub.F @ y)
    sv = float((d * d) @ sub.D)
    return fv / (fv + sv) if fv + sv > 0 else float("nan")


def top_exposures(sub: Sub, d: np.ndarray, k: int = 3) -> list:
    """팩터 몫 경보 날의 큰 노출 셋(PO-5.3·PO-7 MINOR 7): 능동 노출 y = Xᵀd 를 팩터 변동성 √F_kk 로 잰 크기 순."""
    y = sub.X.T @ d
    sz = np.abs(y) * np.sqrt(np.clip(np.diag(sub.F), 0.0, None))
    order = np.argsort(-sz)[:k]
    return [(str(sub.names[j]), float(y[j])) for j in order]
