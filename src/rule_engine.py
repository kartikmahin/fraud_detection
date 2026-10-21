"""
Rule Engine for Multi-Type Fraud Detection

Evaluates rule-based signals across three categories:
  1. IP Anomaly  — suspicious IP patterns (foreign IP, VPN, proxy, multi-account, geo-distance)
  2. Small-Amount Fraud — card testing / rapid small transactions
  3. Velocity — high transaction frequency in 24h

Each rule returns a Signal dict with: category, rule_id, severity, weight, explanation.
Rules are skipped when the relevant fields are absent (None).
"""

from typing import Any, Dict, List


# Severity thresholds
HIGH_SEVERITY = 0.8
MED_SEVERITY = 0.5
LOW_SEVERITY = 0.3


def _make_signal(category: str, rule_id: str, severity: float, weight: float, explanation: str) -> Dict[str, Any]:
    return {
        "category": category,
        "rule_id": rule_id,
        "severity": round(severity, 3),
        "weight": round(weight, 3),
        "explanation": explanation,
    }


class RuleEngine:
    """Stateless rule engine for multi-type fraud detection."""

    def evaluate_all(self, transaction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate all rules and return a list of triggered signals."""
        signals: List[Dict[str, Any]] = []

        # 1. IP Anomaly rules
        signals.extend(self._ip_rules(transaction))

        # 2. Small-amount pattern rules
        signals.extend(self._small_amount_rules(transaction))

        # 3. Velocity rules
        signals.extend(self._velocity_rules(transaction))

        return signals

    # ------------------------------------------------------------------
    # IP Anomaly rules
    # ------------------------------------------------------------------
    def _ip_rules(self, txn: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals = []

        ip_country = txn.get("ip_country")
        ip_is_vpn = txn.get("ip_is_vpn")
        ip_is_proxy = txn.get("ip_is_proxy")
        ip_login_count = txn.get("ip_login_count")
        ip_geo_distance = txn.get("ip_geo_distance_km")

        # Foreign IP
        if ip_country is not None and ip_country not in ("IN", "India"):
            signals.append(_make_signal(
                "ip_anomaly", "foreign_ip", 0.6, 0.25,
                f"Transaction from foreign IP country '{ip_country}' — outside registered region."
            ))

        # VPN usage
        if ip_is_vpn:
            signals.append(_make_signal(
                "ip_anomaly", "vpn_detected", 0.5, 0.15,
                "Connection via VPN detected — may indicate location spoofing."
            ))

        # Proxy usage
        if ip_is_proxy:
            signals.append(_make_signal(
                "ip_anomaly", "proxy_detected", 0.5, 0.15,
                "Connection via proxy detected — may indicate anonymization."
            ))

        # Multi-account on same IP
        if ip_login_count is not None and ip_login_count >= 3:
            sev = min(0.9, 0.3 + (ip_login_count - 3) * 0.1)
            signals.append(_make_signal(
                "ip_anomaly", "multi_account_ip", sev, 0.20,
                f"{ip_login_count} different accounts on same IP in 24h — possible account farm."
            ))

        # Large geo-distance
        if ip_geo_distance is not None and ip_geo_distance >= 500:
            sev = min(0.9, 0.4 + (ip_geo_distance / 5000))
            signals.append(_make_signal(
                "ip_anomaly", "geo_distance", sev, 0.20,
                f"IP geo distance {ip_geo_distance:.0f} km from registered city — unusual location."
            ))

        # Combined IP anomaly: VPN + foreign + multi-account
        if ip_is_vpn and ip_country not in ("IN", "India", None) and (ip_login_count or 0) >= 3:
            signals.append(_make_signal(
                "ip_anomaly", "ip_anomaly_combo", 0.85, 0.30,
                "Combined IP anomaly: VPN + foreign IP + multi-account — high risk."
            ))

        return signals

    # ------------------------------------------------------------------
    # Small-amount pattern rules
    # ------------------------------------------------------------------
    def _small_amount_rules(self, txn: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals = []

        small_count = txn.get("small_txn_count_6h")
        cum_amount = txn.get("cumulative_amount_6h")
        same_merchant = txn.get("same_merchant_count_1h")
        txn_amount = txn.get("transaction_amount")

        # High small-txn count in 6h
        if small_count is not None and small_count >= 5:
            sev = min(0.9, 0.4 + (small_count - 5) * 0.05)
            signals.append(_make_signal(
                "small_amount", "high_small_txn_count", sev, 0.25,
                f"{small_count} small transactions (< ₹200) in 6 hours — possible card testing."
            ))

        # Cumulative small amount buildup
        if cum_amount is not None and cum_amount >= 500:
            sev = min(0.8, 0.3 + (cum_amount - 500) / 2000)
            signals.append(_make_signal(
                "small_amount", "cumulative_small_amount", sev, 0.20,
                f"Cumulative small-transaction total ₹{cum_amount:.0f} in 6h — possible smurfing."
            ))

        # Same merchant repetition
        if same_merchant is not None and same_merchant >= 3:
            sev = min(0.7, 0.3 + (same_merchant - 3) * 0.1)
            signals.append(_make_signal(
                "small_amount", "same_merchant_repeat", sev, 0.15,
                f"{same_merchant} transactions to same merchant in 1 hour — possible card testing."
            ))

        # Small amount but high frequency
        if txn_amount is not None and txn_amount < 200 and (txn.get("transaction_count_24h") or 0) >= 10:
            signals.append(_make_signal(
                "small_amount", "small_amount_high_freq", 0.6, 0.20,
                f"Small amount ₹{txn_amount:.0f} with {txn.get('transaction_count_24h')} txns in 24h — card testing pattern."
            ))

        # Combo: small count + same merchant + VPN
        if (small_count is not None and small_count >= 5 and
            same_merchant is not None and same_merchant >= 3 and
            txn.get("ip_is_vpn")):
            signals.append(_make_signal(
                "small_amount", "small_amount_combo", 0.9, 0.30,
                "Combined small-amount fraud: high small-txn count + merchant repetition + VPN."
            ))

        return signals

    # ------------------------------------------------------------------
    # Velocity rules
    # ------------------------------------------------------------------
    def _velocity_rules(self, txn: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals = []

        txn_count = txn.get("transaction_count_24h")
        acct_age = txn.get("account_age_days")

        if txn_count is not None and txn_count >= 15:
            signals.append(_make_signal(
                "velocity", "extreme_frequency", 0.8, 0.25,
                f"{txn_count} transactions in 24h — extremely high frequency."
            ))
        elif txn_count is not None and txn_count >= 10:
            signals.append(_make_signal(
                "velocity", "high_frequency", 0.5, 0.15,
                f"{txn_count} transactions in 24h — above normal patterns."
            ))

        # New account + high velocity
        if acct_age is not None and acct_age < 90 and txn_count is not None and txn_count >= 10:
            signals.append(_make_signal(
                "velocity", "new_account_velocity", 0.75, 0.20,
                f"New account ({acct_age} days) with {txn_count} txns in 24h — account takeover risk."
            ))

        return signals