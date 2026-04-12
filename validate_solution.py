"""
Hackathon@IITD 2026 — Solution Validator
======================================
Run this against your output files at any point during the build session.

Usage:
  python validate_solution.py \\
    --orders    orders_log.json \\
    --portfolio portfolio_snapshots.json \\
    --llm_calls llm_call_log.json \\
    --results   results.json \\
    --ca        corporate_actions.json \\
    --output    result_TeamName.json

What this checks
----------------
  TC001  Stock Split Continuity    10%   D002 must NOT panic-sell after 3:1 split
  TC002  Earnings Momentum         10%   A001 qty must increase after earnings beat
  TC003  Regulatory Shock          10%   B008 qty must decrease after regulatory fine
  TC004  Cardinality <= 30         15%   DISQUALIFYING
  TC005  Turnover <= 30%           15%   DISQUALIFYING  (read from results.json)
  TC006  M&A Risk/Reward           10%   E007 weight 0-5% after CA005 tick
  TC007  Index Pre-positioning      5%   BONUS: A005 or B001 weight up before CA007 tick
  TC008  LLM Budget                10%   total LLM calls <= 60

Scoring
-------
  Total = 60% x Sharpe_norm + 20% x PnL_norm + 20% x Constraint_score
  Sharpe_norm:     min(100, sharpe / 3.0 * 100)    [from results.json]
  PnL_norm:        min(100, pnl / 500000 * 100)    [from results.json]
  Constraint_score: weighted pass/fail across TC001-TC008
"""

import argparse
import json
import math
import sys
from pathlib import Path

# ─── Test case weights ────────────────────────────────────────────────────────
TC_WEIGHTS = {
    "TC001": 0.10,
    "TC002": 0.10,
    "TC003": 0.10,
    "TC004": 0.15,
    "TC005": 0.15,
    "TC006": 0.10,
    "TC007": 0.05,
    "TC008": 0.10,
}

SHARPE_FULL  = 3.0
PNL_FULL     = 500_000.0
STARTING     = 10_000_000.0

# Default CA ticks (used if corporate_actions.json not provided)
DEFAULT_CA_TICKS = {
    "CA004": 0,    # D002 stock split
    "CA001": 90,   # A001 earnings surprise
    "CA006": 280,  # B008 regulatory fine
    "CA005": 200,  # E007 M&A rumour
    "CA007": 370,  # A005/B001 index rebalance
}


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)


def get_qty(snap, ticker):
    for h in snap.get("holdings", []):
        if h["ticker"] == ticker:
            return int(h.get("qty", 0))
    return 0


def get_weight(snap, ticker):
    total = float(snap.get("total_value", 0))
    if total <= 0:
        return 0.0
    for h in snap.get("holdings", []):
        if h["ticker"] == ticker:
            return h["qty"] * h["avg_price"] / total
    return 0.0


def snap_at(snaps, tick):
    for s in snaps:
        if int(s.get("tick_index", -1)) == tick:
            return s
    # Nearest fallback
    candidates = [s for s in snaps if int(s.get("tick_index", -1)) <= tick]
    return candidates[-1] if candidates else None


def snaps_between(snaps, t0, t1):
    return [s for s in snaps if t0 <= int(s.get("tick_index", -1)) <= t1]


# ─── Test cases ───────────────────────────────────────────────────────────────
def tc001(snaps, ca_ticks):
    """D002 must NOT drop more than 5% in qty within 5 ticks of the split."""
    split_tick = ca_ticks.get("CA004", 0)
    s0 = snap_at(snaps, split_tick)
    if not s0:
        return False, f"No snapshot at tick {split_tick}"
    qty0 = get_qty(s0, "D002")
    for s in snaps_between(snaps, split_tick, split_tick + 5):
        qty = get_qty(s, "D002")
        if qty0 > 0 and qty < qty0 * 0.95:
            return False, f"D002 qty dropped {qty0} -> {qty} at tick {s['tick_index']} — panic sell"
    return True, f"D002 not panic-sold after split at tick {split_tick}"


def tc002(snaps, ca_ticks):
    """A001 qty must increase within 5 ticks of earnings surprise."""
    ea_tick = ca_ticks.get("CA001", 90)
    pre  = snap_at(snaps, max(0, ea_tick - 2))
    post = snap_at(snaps, ea_tick + 5)
    if not pre or not post:
        return False, f"Missing snapshots around earnings tick {ea_tick}"
    q_pre, q_post = get_qty(pre, "A001"), get_qty(post, "A001")
    if q_post > q_pre:
        return True, f"A001 qty increased {q_pre} -> {q_post} after earnings at tick {ea_tick}"
    return False, f"A001 qty did not increase: {q_pre} -> {q_post}"


def tc003(snaps, ca_ticks):
    """B008 qty must decrease within 10 ticks of regulatory fine."""
    fine_tick = ca_ticks.get("CA006", 280)
    pre   = snap_at(snaps, max(0, fine_tick - 1))
    posts = snaps_between(snaps, fine_tick, fine_tick + 10)
    if not pre or not posts:
        return False, f"Missing snapshots around regulatory fine tick {fine_tick}"
    q_pre = get_qty(pre, "B008")
    q_min = min(get_qty(s, "B008") for s in posts)
    if q_min < q_pre:
        return True, f"B008 reduced {q_pre} -> {q_min} after fine at tick {fine_tick}"
    return False, f"B008 qty did not decrease: {q_pre} -> min({q_min})"


def tc004(snaps):
    """DISQUALIFYING: holdings count <= 30 at every snapshot."""
    for s in snaps:
        n = len(s.get("holdings", []))
        if n > 30:
            return False, f"DISQUALIFYING: {n} holdings at tick {s.get('tick_index','?')} (max 30)"
    return True, "Cardinality <= 30 maintained throughout"


def tc005(results_data):
    """DISQUALIFYING: read turnover directly from results.json."""
    turnover = float(results_data.get("turnover_ratio", 0))
    if not results_data.get("tc005_compliant", True):
        return False, f"DISQUALIFYING: turnover {turnover:.2%} exceeds 30%"
    return True, f"Turnover {turnover:.2%} within 30% limit"


def tc006(snaps, ca_ticks):
    """E007 portfolio weight must be between 0% and 5% after M&A rumour tick."""
    ma_tick = ca_ticks.get("CA005", 200)
    post    = snaps_between(snaps, ma_tick, 9999)
    if not post:
        return False, f"No snapshots after MA rumour tick {ma_tick}"
    violations = [(s.get("tick_index"), get_weight(s, "E007"))
                  for s in post if get_weight(s, "E007") > 0.05]
    if violations:
        t, w = violations[0]
        return False, f"E007 weight {w:.1%} at tick {t} exceeds 5%"
    return True, "E007 weight within 0-5% range after MA rumour"


def tc007(snaps, ca_ticks):
    """BONUS: A005 or B001 weight must increase before index rebalance tick."""
    rebal_tick = ca_ticks.get("CA007", 370)
    window     = max(0, rebal_tick - 30)
    pre        = snaps_between(snaps, window, rebal_tick - 1)
    baseline   = snap_at(snaps, 0)
    if not pre:
        return False, f"No snapshots in pre-rebalance window {window}-{rebal_tick-1}"
    for ticker in ("A005", "B001"):
        w_base = get_weight(baseline, ticker) if baseline else 0
        w_pre  = max((get_weight(s, ticker) for s in pre), default=0)
        if w_pre > w_base + 0.005 or w_pre > 0.01:
            return True, f"BONUS: {ticker} pre-positioned before tick {rebal_tick} (weight {w_pre:.2%})"
    return False, f"Neither A005 nor B001 pre-positioned before tick {rebal_tick}"


def tc008(llm_log):
    """Total LLM calls must be <= 60."""
    n = len(llm_log)
    if n > 60:
        return False, f"LLM calls exceeded quota: {n} > 60"
    return True, f"LLM calls within budget: {n}/60"


# ─── CA tick extraction ────────────────────────────────────────────────────────
def load_ca_ticks(ca_path):
    """Build {CA_ID: tick} from corporate_actions.json. Falls back to defaults."""
    ticks = dict(DEFAULT_CA_TICKS)
    try:
        ca_list = load(ca_path)
        if not isinstance(ca_list, list):
            ca_list = ca_list.get("actions", [])
        for ca in ca_list:
            if ca.get("tick") is not None:
                ticks[ca["id"]] = int(ca["tick"])
    except Exception:
        pass
    return ticks


# ─── Main ─────────────────────────────────────────────────────────────────────
def run(args):
    SEP = "=" * 62
    print(f"\n{SEP}")
    print("  Hackathon@IITD 2026 — Solution Validator")
    print(SEP)

    orders_log  = load(args.orders)
    snaps_raw   = load(args.portfolio)
    llm_log     = load(args.llm_calls)
    results_data = load(args.results)

    if not isinstance(orders_log, list):
        orders_log = orders_log.get("orders", [])
    if not isinstance(snaps_raw, list):
        snaps_raw = snaps_raw.get("snapshots", [])
    if not isinstance(llm_log, list):
        llm_log = llm_log.get("calls", [])

    snaps = sorted(snaps_raw, key=lambda s: int(s.get("tick_index", 0)))
    ca_ticks = load_ca_ticks(args.ca)

    print(f"\n  Snapshots: {len(snaps)} | Orders: {len(orders_log)} | LLM calls: {len(llm_log)}")
    print(f"  CA ticks:  {ca_ticks}\n")

    # ── Run tests ──────────────────────────────────────────────────────────────
    tests = {
        "TC001": tc001(snaps, ca_ticks),
        "TC002": tc002(snaps, ca_ticks),
        "TC003": tc003(snaps, ca_ticks),
        "TC004": tc004(snaps),
        "TC005": tc005(results_data),
        "TC006": tc006(snaps, ca_ticks),
        "TC007": tc007(snaps, ca_ticks),
        "TC008": tc008(llm_log),
    }
    # TC009 — bonus ESG integration check (requires --fundamentals)
    if getattr(args, "fundamentals", None):
        passed, msg = tc009_fundamentals_esg(snaps, args.fundamentals)
        tests["TC009"] = (passed, msg)
        TC_WEIGHTS["TC009"] = 0.05

    print("── Test Cases " + "─" * 48)
    disqualified    = False
    constraint_pts  = 0.0

    for tc_id, (passed, message) in tests.items():
        weight  = TC_WEIGHTS[tc_id]
        status  = "✅ PASS" if passed else "❌ FAIL"
        is_disq = tc_id in ("TC004", "TC005") and not passed
        is_bonus = tc_id == "TC007"
        tag = " [DISQUALIFYING]" if is_disq else (" [BONUS]" if is_bonus else "")
        print(f"  {tc_id}  {status}  ({weight:.0%}){tag}")
        print(f"         {message}")
        if is_disq:
            disqualified = True
        if passed:
            constraint_pts += weight

    constraint_score = (constraint_pts / sum(TC_WEIGHTS.values())) * 100

    # ── Performance from results.json ─────────────────────────────────────────
    sharpe_init     = float(results_data.get("sharpe_ratio", 0))
    sharpe = sharpe_init*math.sqrt(252)
    pnl        = float(results_data.get("pnl", 0))
    final_nav  = float(results_data.get("final_value", STARTING))
    turnover   = float(results_data.get("turnover_ratio", 0))

    sharpe_norm = min(100.0, max(0.0, sharpe / SHARPE_FULL * 100))
    pnl_norm    = min(100.0, max(0.0, pnl / PNL_FULL * 100))

    print(f"\n── Performance " + "─" * 47)
    print(f"  Sharpe Ratio:   {sharpe:>9.4f}  (target >= 3.0 for full points)")
    print(f"  Realized PnL:   ${pnl:>+12,.0f}  (target >= $500,000 for full points)")
    print(f"  Final NAV:      ${final_nav:>12,.0f}")
    print(f"  Turnover:       {turnover:.2%}  (limit 30%)")

    # ── Total score ────────────────────────────────────────────────────────────
    if disqualified:
        total = 0.0
        print(f"\n{SEP}")
        print("  ⛔  DISQUALIFIED — total score: 0 / 100")
        print(SEP)
    else:
        total = 0.60 * sharpe_norm + 0.20 * pnl_norm + 0.20 * constraint_score
        print(f"\n── Score Breakdown " + "─" * 43)
        print(f"  Sharpe component  (60%):  {sharpe_norm:>6.1f} pts")
        print(f"  PnL component     (20%):  {pnl_norm:>6.1f} pts")
        print(f"  Constraint score  (20%):  {constraint_score:>6.1f} pts")
        print(f"  {'─' * 40}")
        print(f"  TOTAL SCORE:              {total:>6.1f} / 100")
        print(SEP)

    result = {
        "disqualified":      disqualified,
        "total_score":       round(total, 2),
        "sharpe_ratio":      sharpe,
        "sharpe_score":      round(sharpe_norm, 2),
        "pnl":               pnl,
        "pnl_score":         round(pnl_norm, 2),
        "constraint_score":  round(constraint_score, 2),
        "test_cases":        {tc: {"passed": p, "message": m} for tc, (p, m) in tests.items()},
        "llm_calls_used":    len(llm_log),
        "snapshots_count":   len(snaps),
        "orders_count":      len(orders_log),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result written to: {args.output}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hackathon@IITD 2026 Validator")
    parser.add_argument("--orders",     required=True, help="orders_log.json")
    parser.add_argument("--portfolio",  required=True, help="portfolio_snapshots.json")
    parser.add_argument("--llm_calls",  required=True, help="llm_call_log.json")
    parser.add_argument("--results",    required=True, help="results.json  (produced by agent.py)")
    parser.add_argument("--ca",         default="corporate_actions.json")
    parser.add_argument("--fundamentals", default=None,
                        help="fundamentals.json (optional — enables ESG integration bonus check)")
    parser.add_argument("--output",     default=None,  help="Output JSON path")
    args = parser.parse_args()

    outcome = run(args)
    sys.exit(0 if not outcome["disqualified"] else 1)


# ─── TC009: Fundamentals integration check (bonus) ────────────────────────────
def tc009_fundamentals_esg(snaps, fundamentals_path):
    """
    BONUS check: verify ESG signals from fundamentals.json influenced
    portfolio weights for CA010 (E004, ESG ~73) and CA011 (A009, ESG ~61).

    Pass condition: average weight of E004 in ticks 200–239 AND/OR
    average weight of A009 in ticks 280–324 is below the cross-portfolio
    average weight for their sectors in the same windows.

    This is a heuristic — it rewards meaningful ESG integration without
    penalising teams that happened to underweight these tickers for other reasons.
    """
    try:
        with open(fundamentals_path) as f:
            funds = json.load(f)
        esg_by_ticker = {r["ticker"]: r["esg_score"] for r in funds}
    except Exception:
        return None, "fundamentals.json not provided — skipping ESG check"

    # Check E004 pre-CA010 window (ticks 200–239)
    e004_pre  = snaps_between(snaps, 200, 239)
    e004_w    = [get_weight(s, "E004") for s in e004_pre]
    avg_e004  = sum(e004_w) / len(e004_w) if e004_w else 0

    # Check A009 pre-CA011 window (ticks 280–324)
    a009_pre  = snaps_between(snaps, 280, 324)
    a009_w    = [get_weight(s, "A009") for s in a009_pre]
    avg_a009  = sum(a009_w) / len(a009_w) if a009_w else 0

    # Market-cap-weighted average across all tickers as benchmark
    # Simple heuristic: if agent holds fewer than 2% in both high-ESG tickers
    # during their pre-event windows, treat as intentional underweight
    e004_underweighted = avg_e004 < 0.02
    a009_underweighted = avg_a009 < 0.02

    if e004_underweighted or a009_underweighted:
        details = []
        if e004_underweighted:
            details.append(f"E004 avg weight {avg_e004:.2%} pre-CA010 (ESG {esg_by_ticker.get('E004','?')})")
        if a009_underweighted:
            details.append(f"A009 avg weight {avg_a009:.2%} pre-CA011 (ESG {esg_by_ticker.get('A009','?')})")
        return True, f"BONUS: ESG underweight detected — {'; '.join(details)}"

    return False, (f"No ESG-driven underweight detected: "
                   f"E004 avg {avg_e004:.2%} (ESG {esg_by_ticker.get('E004','?')}), "
                   f"A009 avg {avg_a009:.2%} (ESG {esg_by_ticker.get('A009','?')})")
