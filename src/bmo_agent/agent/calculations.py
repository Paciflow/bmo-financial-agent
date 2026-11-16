from __future__ import annotations

from typing import Literal

Operation = Literal["percent_change", "percentage_point_change", "difference", "ratio", "sum"]


def compute(operation: Operation, a: float, b: float) -> dict:
    """Deterministic financial arithmetic -- the LLM never computes these itself.
    It only identifies which figures and operation apply; Python does the math.
    """
    if operation == "percent_change":
        if b == 0:
            return {"result": None, "unit": "%", "formula": f"({a} - {b}) / {b} * 100", "error": "division by zero"}
        result = (a - b) / b * 100
        return {"result": round(result, 2), "unit": "%", "formula": f"({a} - {b}) / {b} * 100"}
    if operation == "percentage_point_change":
        return {"result": round(a - b, 2), "unit": "percentage points (pp)", "formula": f"{a} - {b}"}
    if operation == "difference":
        return {"result": round(a - b, 2), "unit": None, "formula": f"{a} - {b}"}
    if operation == "ratio":
        if b == 0:
            return {"result": None, "unit": "x", "formula": f"{a} / {b}", "error": "division by zero"}
        return {"result": round(a / b, 4), "unit": "x", "formula": f"{a} / {b}"}
    if operation == "sum":
        return {"result": round(a + b, 2), "unit": None, "formula": f"{a} + {b}"}
    raise ValueError(f"Unsupported operation: {operation}")
