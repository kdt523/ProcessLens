import pytest

from processlens.agent.schemas import Finding, RootCauseReport, SupportingNumber, ToolCall
from processlens.agent.verifier import matches, parse_numbers, verify

CFG = {"verifier": {"rel_tolerance": 0.01, "abs_tolerance": 0.001}}

LOG = [
    ToolCall(
        id="call_1",
        tool="get_data_health",
        args={},
        output={"runs": 940, "failures": 76, "base_rate": 0.0808510638},
    ),
    ToolCall(
        id="call_2",
        tool="rank_suspects",
        args={"top_k": 5},
        output={
            "suspects": [
                {
                    "consensus_rank": 1,
                    "cluster": 12,
                    "representative": "sensor_060",
                    "q_value": 0.000253046,
                    "effect_size": 0.3975152,
                    "stability_freq": 0.81,
                    "shap_share": 0.0199,
                    "evidence": "strong",
                },
                {
                    "consensus_rank": 2,
                    "cluster": 40,
                    "representative": "sensor_132",
                    "q_value": 0.8695,
                    "effect_size": 0.0424,
                    "stability_freq": 0.0,
                    "shap_share": 0.0422,
                    "evidence": "weak",
                },
            ],
            "evidence_counts": {"strong": 1, "weak": 1},
        },
    ),
    ToolCall(
        id="call_3",
        tool="get_known_limits",
        args={},
        output={"limits": ["Linear faults: found once β = 1 (94% at β = 2)."]},
    ),
]


def good_report(**overrides) -> RootCauseReport:
    base = {
        "summary": "One suspect cluster is associated with failures; base rate 8.1%.",
        "findings": [
            Finding(
                cluster=12,
                representative_sensor="sensor_060",
                evidence_strength="strong",
                supporting_numbers=[
                    SupportingNumber(name="q_value", value=0.000253, tool_call_id="call_2"),
                    SupportingNumber(name="effect_size", value=0.398, tool_call_id="call_2"),
                ],
                onset_date="2008-08-21",
                interpretation="Failing runs read higher on sensor_060 since 2008-08-21.",
            )
        ],
        "recommended_checks": ["Inspect sensor_060 against maintenance logs."],
        "limits": ["Linear faults: found once β = 1 (94% at β = 2)."],
        "decision": "investigate",
    }
    base.update(overrides)
    return RootCauseReport(**base)


def test_good_report_passes() -> None:
    v = verify(good_report(), LOG, CFG)
    assert v.ok, v.problems
    assert v.faithfulness == 1.0
    assert v.n_numbers >= 4


def test_deliberately_wrong_number_is_caught() -> None:
    rep = good_report()
    rep.findings[0].supporting_numbers[1].value = 0.55  # tool said 0.3975
    v = verify(rep, LOG, CFG)
    assert not v.ok
    assert any("effect_size=0.55" in p for p in v.problems)
    assert v.faithfulness < 1.0


def test_invented_number_in_free_text_is_caught() -> None:
    v = verify(good_report(summary="Failure risk rises 37% on this sensor."), LOG, CFG)
    assert not v.ok and any("37" in p for p in v.problems)


def test_wrong_citation_is_caught() -> None:
    rep = good_report()
    rep.findings[0].supporting_numbers[0].tool_call_id = "call_1"  # number is in call_2
    assert not verify(rep, LOG, CFG).ok
    rep.findings[0].supporting_numbers[0].tool_call_id = "call_99"
    assert any("unknown tool call" in p for p in verify(rep, LOG, CFG).problems)


def test_upgraded_evidence_is_caught() -> None:
    rep = good_report()
    rep.findings[0].cluster = 40
    rep.findings[0].representative_sensor = "sensor_132"
    rep.findings[0].supporting_numbers = [
        SupportingNumber(name="q_value", value=0.8695, tool_call_id="call_2")
    ]
    v = verify(rep, LOG, CFG)
    assert any("engine graded cluster 40 'weak'" in p for p in v.problems)


def test_causal_language_is_caught_but_negation_allowed() -> None:
    assert not verify(good_report(summary="sensor_060 causes the failures."), LOG, CFG).ok
    ok = verify(
        good_report(summary="This does not prove that the sensor causes failures."), LOG, CFG
    )
    assert ok.ok, ok.problems


def test_decision_consistency() -> None:
    v = verify(good_report(findings=[], decision="investigate"), LOG, CFG)
    assert any("no findings" in p for p in v.problems)


def test_unknown_sensor_is_caught() -> None:
    rep = good_report()
    rep.findings[0].representative_sensor = "sensor_999"
    assert any("sensor_999" in p for p in verify(rep, LOG, CFG).problems)


def test_parse_numbers_skips_ids_and_dates_and_handles_percent() -> None:
    nums = parse_numbers("sensor_060 shifted on 2008-08-21 (call_2): 8.1% vs 0.25 and -1")
    assert [round(v, 4) for v, _ in nums] == [0.081, 0.25, -1.0]


@pytest.mark.parametrize(
    ("value", "dec", "ok"),
    [
        (0.398, 3, True),
        (0.40, 2, True),
        (0.4, 1, True),
        (0.39, 2, False),
        (0.3975152, None, True),
    ],
)
def test_rounding_aware_matching(value, dec, ok) -> None:
    assert matches(value, [0.3975152], dec, 0.01, 0.001) is ok
