"""
Fraud Signal Fusion Engine

Combines ML model probability with rule-based signals into a single
composite risk score. Uses a weighted fusion approach:

  final_score = ml_weight * ml_score + (1 - ml_weight) * rule_score

where rule_score is derived from the weighted average of triggered rule
severities. The fusion also produces a detailed breakdown of all signals
(triggered and non-triggered) for transparency.
"""

from typing import Any, Dict, List


# Risk level thresholds
RISK_LEVELS = [
    (0.8, "CRITICAL"),
    (0.6, "HIGH"),
    (0.3, "MEDIUM"),
    (0.0, "LOW"),
]


def _risk_level(score: float) -> str:
    for threshold, level in RISK_LEVELS:
        if score >= threshold:
            return level
    return "LOW"


class FraudSignalFusion:
    """Fuses ML probability with rule-based signals into a composite risk score."""

    def __init__(self, ml_weight: float = 0.6):
        self.ml_weight = ml_weight
        self.rule_weight = 1.0 - ml_weight

    def fuse(
        self,
        ml_probability: float,
        rule_signals: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> Any:
        """Fuse ML score with rule signals.

        Args:
            ml_probability: ML model's fraud probability (0-1).
            rule_signals: List of triggered rule signals from RuleEngine.
            threshold: Classification threshold.

        Returns:
            A FusedResult object with all fields.
        """
        # Compute rule score as weighted average of triggered signals
        if rule_signals:
            total_weight = sum(s["weight"] for s in rule_signals)
            if total_weight > 0:
                rule_score = sum(s["severity"] * s["weight"] for s in rule_signals) / total_weight
            else:
                rule_score = 0.0
        else:
            rule_score = 0.0

        # Cap rule score at 1.0
        rule_score = min(1.0, rule_score)

        # Fuse
        final_score = self.ml_weight * ml_probability + self.rule_weight * rule_score
        final_score = min(1.0, max(0.0, final_score))

        # Determine fraud type indicators from triggered signals
        fraud_type_indicators = []
        categories = set(s["category"] for s in rule_signals)
        if "ip_anomaly" in categories:
            fraud_type_indicators.append("ip_anomaly")
        if "small_amount" in categories:
            fraud_type_indicators.append("small_amount")
        if "velocity" in categories:
            fraud_type_indicators.append("velocity")
        if not fraud_type_indicators and final_score >= threshold:
            fraud_type_indicators.append("ml_detected")

        # Positive signals (non-triggered rules) — we don't have them from evaluate_all
        # but we can generate them based on absence of risk signals
        positive_signals = self._generate_positive_signals(rule_signals)

        return FusedResult(
            is_fraud=final_score >= threshold,
            final_risk_score=round(final_score, 4),
            risk_level=_risk_level(final_score),
            ml_score=round(ml_probability, 4),
            rule_score=round(rule_score, 4),
            fraud_type_indicators=fraud_type_indicators,
            signals=rule_signals,
            positive_signals=positive_signals,
            threshold=threshold,
        )


    def _generate_positive_signals(self, triggered_signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate positive (non-triggered) signal entries for transparency."""
        positive = []
        triggered_ids = set(s["rule_id"] for s in triggered_signals)

        # If no IP anomaly signals triggered, note that IP looks normal
        if not any(s["category"] == "ip_anomaly" for s in triggered_signals):
            positive.append({
                "category": "ip_anomaly",
                "rule_id": "ip_clean",
                "severity": 0.0,
                "weight": 0.1,
                "explanation": "No IP anomalies detected — IP profile appears normal.",
            })

        # If no small-amount signals triggered
        if not any(s["category"] == "small_amount" for s in triggered_signals):
            positive.append({
                "category": "small_amount",
                "rule_id": "no_small_amount_pattern",
                "severity": 0.0,
                "weight": 0.1,
                "explanation": "No small-amount fraud pattern detected.",
            })

        # If no velocity signals triggered
        if not any(s["category"] == "velocity" for s in triggered_signals):
            positive.append({
                "category": "velocity",
                "rule_id": "normal_velocity",
                "severity": 0.0,
                "weight": 0.1,
                "explanation": "Transaction frequency within normal range.",
            })

        return positive


class FusedResult:
    """Container for fused analysis result."""

    def __init__(
        self,
        is_fraud: bool,
        final_risk_score: float,
        risk_level: str,
        ml_score: float,
        rule_score: float,
        fraud_type_indicators: List[str],
        signals: List[Dict[str, Any]],
        positive_signals: List[Dict[str, Any]],
        threshold: float,
    ):
        self.is_fraud = is_fraud
        self.final_risk_score = final_risk_score
        self.risk_level = risk_level
        self.ml_score = ml_score
        self.rule_score = rule_score
        self.fraud_type_indicators = fraud_type_indicators
        self.signals = signals
        self.positive_signals = positive_signals
        self.threshold = threshold