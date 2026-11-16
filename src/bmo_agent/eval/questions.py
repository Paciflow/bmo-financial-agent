"""20 manually verified evaluation questions against BMO's 2025 MD&A.

Every `expected_pages` value was confirmed by manually reading the corresponding
page of data/processed/raw_pages.json before being written here -- these are not
LLM-generated guesses. Pages are 1-based (as printed in the PDF / cited to users).

Note on the "chart/diagram" category: this PDF has
zero embedded *raster* images, but it does contain vector-drawn diagrams (BMO's
operating-segments org chart, page 20) whose 2D box structure is lost by native
text extraction. q14 targets that page; see interpretation/diagrams.py and
scripts/interpret_diagrams.py for the Claude-vision pass that recovers it, and
README's "Known limitations" section for how that page was identified and why
auto-detecting every such page on a larger document is still future work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Category = Literal[
    "narrative",
    "table",
    "reported_vs_adjusted",
    "yoy_comparison",
    "operating_segment",
    "org_structure",  # stand-in for "chart/diagram" -- see module docstring
    "ambiguous",
    "general",
    "unsupported",
]


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: Category
    expected_pages: list[int] | None = None  # 1-based; None for ambiguous/general
    expected_values: dict | None = None  # for numerical-accuracy checks
    expect_clarification: bool = False
    expect_citations: bool = True


QUESTIONS: list[EvalQuestion] = [
    EvalQuestion(
        id="q01",
        question="What was BMO's total revenue in fiscal 2025 compared to fiscal 2024, on a reported basis?",
        category="table",
        expected_pages=[10],
        expected_values={"revenue_2025": 36274, "revenue_2024": 32795},
    ),
    EvalQuestion(
        id="q02",
        question="What is the difference between BMO's reported net income and adjusted net income for fiscal 2025?",
        category="reported_vs_adjusted",
        expected_pages=[10],
        expected_values={"reported_net_income_2025": 8725, "adjusted_net_income_2025": 9248},
    ),
    EvalQuestion(
        id="q03",
        question="What was BMO's reported diluted EPS versus adjusted diluted EPS for fiscal 2025?",
        category="reported_vs_adjusted",
        expected_pages=[10],
        expected_values={"reported_eps_2025": 11.44, "adjusted_eps_2025": 12.16},
    ),
    EvalQuestion(
        id="q04",
        question="By what percentage did BMO's reported revenue grow from fiscal 2024 to fiscal 2025?",
        category="yoy_comparison",
        expected_pages=[10],
        expected_values={"percent_change": 10.61},
    ),
    EvalQuestion(
        id="q05",
        question="What was the reported net income of the Canadian Personal and Commercial Banking segment in fiscal 2025 compared to fiscal 2024?",
        category="operating_segment",
        expected_pages=[23],
        expected_values={"net_income_2025": 3295, "net_income_2024": 3457},
    ),
    EvalQuestion(
        id="q06",
        question="How many percentage points did Canadian P&C's return on equity change from fiscal 2024 to fiscal 2025?",
        category="operating_segment",
        expected_pages=[23],
        expected_values={"roe_2025": 19.4, "roe_2024": 21.4, "pp_change": -2.0},
    ),
    EvalQuestion(
        id="q07",
        question="What lines of business make up BMO's U.S. Banking segment?",
        category="narrative",
        expected_pages=[25],
    ),
    EvalQuestion(
        id="q08",
        question="How does BMO define Environmental and Social Risk?",
        category="narrative",
        expected_pages=[91],
    ),
    EvalQuestion(
        id="q09",
        question="What was BMO's CET1 Ratio as at October 31, 2025, compared with October 31, 2024?",
        category="table",
        expected_pages=[47, 48],
        expected_values={"cet1_2025": 13.3, "cet1_2024": 13.6},
    ),
    EvalQuestion(
        id="q10",
        question="By how many percentage points did BMO's CET1 Ratio decline from fiscal 2024 to fiscal 2025?",
        category="yoy_comparison",
        expected_pages=[47],
        expected_values={"pp_change": -0.3},
    ),
    EvalQuestion(
        id="q11",
        question="What is OSFI's minimum CET1 Ratio requirement under the Basel III framework?",
        category="table",
        expected_pages=[45],
        expected_values={"min_cet1": 4.5},
    ),
    EvalQuestion(
        id="q12",
        question="What are the two lines of business within Canadian Personal and Commercial Banking?",
        category="narrative",
        expected_pages=[21],
    ),
    EvalQuestion(
        id="q13",
        question="What was BMO's total outstanding residential mortgage balance in the Ontario region as at October 31, 2025?",
        category="table",
        expected_pages=[67],
        expected_values={"ontario_total": 83001},
    ),
    EvalQuestion(
        id="q14",
        question="What operating segments does BMO report results under?",
        category="org_structure",
        expected_pages=[20],
    ),
    EvalQuestion(
        id="q15",
        # Names a time period ("this year" = fiscal 2025) even without a segment or
        # metric -- a reasonable analyst answers with the enterprise headline
        # figures by default rather than refusing. Was previously (wrongly) treated
        # as ambiguous; see agent/prompts.py CLASSIFY_PROMPT for the fix.
        question="How did we do this year?",
        category="table",
        expected_pages=[8, 10],
        expect_clarification=False,
    ),
    EvalQuestion(
        id="q16",
        # Genuinely scope-free: no year, no segment, no accounting basis. This is
        # the case that should still trigger clarification.
        question="What was the net income?",
        category="ambiguous",
        expect_clarification=True,
        expect_citations=False,
    ),
    EvalQuestion(
        id="q17",
        question="What is a price-to-earnings (P/E) ratio and how is it generally interpreted?",
        category="general",
        expect_citations=False,
    ),
    EvalQuestion(
        id="q18",
        question="What does an inverted yield curve typically signal about the economy?",
        category="general",
        expect_citations=False,
    ),
    EvalQuestion(
        id="q19",
        question="What was BMO's stock closing price on November 15, 2025?",
        category="unsupported",
        expect_citations=False,
    ),
    EvalQuestion(
        id="q20",
        question="What was TD Bank's net income for fiscal 2025?",
        category="unsupported",
        expect_citations=False,
    ),
]
