import math
import datetime as dt
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------
# Helper types & utilities
# ------------------------------
@dataclass
class Person:
    birthdate: dt.date
    start_service_db: dt.date
    today: dt.date

def years_between(d1: dt.date, d2: dt.date) -> float:
    return (d2 - d1).days / 365.25

def add_years(d: dt.date, years: float) -> dt.date:
    days = int(round(years * 365.25))
    return d + dt.timedelta(days=days)

def shift_date(d: dt.date, delta_days: int) -> dt.date:
    return d + dt.timedelta(days=delta_days)

# ------------------------------
# Core helpers
# ------------------------------
def ann_growth_series(start: float, growth: float, n: int) -> List[float]:
    return [start * ((1 + growth) ** i) for i in range(n)]

def accumulate_stream(contribs: List[float], nominal_return: float) -> float:
    """Year-end contributions compounding at nominal_return."""
    bal = 0.0
    for c in contribs:
        bal = bal * (1 + nominal_return) + c
    return bal

# ------------------------------
# DB plan calculations (simplified New Society style)
# ------------------------------
def db_high5_from_salary_series(salaries: List[float]) -> float:
    if len(salaries) < 5:
        return float(np.mean(salaries)) if salaries else 0.0
    best = 0.0
    for i in range(0, len(salaries) - 4):
        five_avg = float(np.mean(salaries[i:i+5]))
        best = max(best, five_avg)
    return best

def calc_db_lifetime_and_bridge(
    high5: float,
    best_avg_ympe: float,
    service_years: float,
    post_2018_service: float,
    accrual_rate: float,
    integration_rate: float,
    bridge_rate: float
) -> Tuple[float, float]:
    accrual = accrual_rate * high5 * service_years
    integration = integration_rate * best_avg_ympe * service_years
    lifetime = accrual - integration
    bridge = bridge_rate * best_avg_ympe * max(0.0, post_2018_service)
    return max(lifetime, 0.0), max(bridge, 0.0)

# ------------------------------
# Streamlit app
# ------------------------------
st.set_page_config(page_title="DB vs DC Retirement Model — Interactive", layout="wide")
st.title("DB vs DC Retirement Model — Interactive")
st.caption("Educational model: simplifies some pension mechanics (YMPE, indexing caps, taxes). Adjust assumptions to explore sensitivity.")

with st.sidebar:
    st.header("Profile & Dates")
    birthdate_in = st.date_input("Birthdate", value=dt.date(1987, 11, 16))
    db_start_in = st.date_input("DB Service Start", value=dt.date(2015, 11, 19))
    today_in = st.date_input("Analysis Today", value=dt.date.today())
    mgr_switch_actual = st.date_input("Manager/DC switch date (actual)", value=dt.date(2025, 8, 1),
                                      help="When you would actually switch to DC (take Manager role).")

    st.header("General Assumptions")
    inflation = st.slider("Inflation (CPI)", 0.0, 0.06, 0.02, 0.005)
    ympe_growth = st.slider("YMPE Growth", 0.0, 0.06, 0.03, 0.005)
    current_ympe = st.number_input("Current YMPE", min_value=0, value=68500, step=500)

    st.header("Salary & STIP")
    base_salary_now = st.number_input("Current Base Salary (DB path)", min_value=0, value=150000, step=1000)
    manager_premium = st.slider("Manager Salary Premium (DC path)", 0.00, 0.50, 0.20, 0.01)
    salary_growth_db = st.slider("Annual Salary Growth — DB (real)", 0.00, 0.05, 0.02, 0.005)
    salary_growth_dc = st.slider("Annual Salary Growth — DC (real)", 0.00, 0.05, 0.025, 0.005)
    stip_share = st.slider("STIP as % of Base (counts up to 50%)", 0.00, 0.50, 0.20, 0.01)

    st.header("DC Contributions & Returns")
    employee_dc = st.slider("Employee DC %", 0.00, 0.06, 0.06, 0.005)
    employer_match = st.slider("Employer Match %", 0.00, 0.06, 0.06, 0.005)
    dc_return_nominal = st.slider("DC Investment Return (nominal)", 0.00, 0.12, 0.07, 0.005)
    fee_drag = st.slider("Fee Drag", 0.00, 0.02, 0.002, 0.001)
    dc_return_net = max(0.0, dc_return_nominal - fee_drag)

    st.header("Side Savings from Premium")
    assume_half_premium_after_tax = st.checkbox("Assume 50% of premium remains after tax", value=True)
    after_tax_multiplier = 0.50 if assume_half_premium_after_tax else st.slider("After-Tax Multiplier", 0.0, 1.0, 0.50, 0.05)
    save_fraction_of_premium = st.slider("Save % of After-Tax Premium", 0.00, 1.00, 0.50, 0.05)
    side_return_nominal = st.slider("Side Savings Return (nominal)", 0.00, 0.12, 0.06, 0.005)

    st.header("Retirement Ages & Govt. Benefits")
    retire_early_age = st.number_input("Early Retirement Age", 40.0, 65.0, 56.5, 0.5)
    retire_normal_age = st.number_input("Normal Retirement Age", 55.0, 70.0, 65.0, 0.5)

    snapshot_mode = st.checkbox(
        "Snapshot mode: Today = Early Eligibility; replay manager switch at same career offset",
        value=False,
        help=("Shifts Birthdate & DB Start so Today acts as early-eligibility date, "
              "and starts DC at the same years-from-DB-start as your actual manager switch.")
    )

    st.header("Rollback (discount) for past years")
    rollback_rate = st.slider("Backward salary discount for past years (deflationary)", 0.0, 0.05, 0.02, 0.001,
                              help="Used to roll back today's salary & premium to earlier years when simulating past contributions.")
    apply_rollback_to_stip = st.checkbox("Apply rollback to STIP base too", value=True)
    apply_rollback_to_side = st.checkbox("Apply rollback to side-savings base too", value=True)

    cpp_at_65_today = st.number_input("CPP @65 (today $)", 0, 16375, 250)
    oas_at_65_today = st.number_input("OAS @65 (today $)", 0, 8560, 100)
    show_cpp_oas = st.checkbox("Overlay CPP & OAS on charts", value=True)

    st.header("DB Parameters")
    accrual_rate = st.slider("DB Accrual Rate", 0.0, 0.025, 0.02, 0.001)
    integration_rate = st.slider("DB YMPE Integration Rate", 0.0, 0.01, 0.00625, 0.00025)
    bridge_rate = st.slider("Bridge Rate (post-2018 service)", 0.0, 0.01, 0.00625, 0.00025)
    index_fraction_cpi = st.slider("DB Indexing as % of CPI", 0.0, 1.0, 0.75, 0.05)

# ---------- Effective dates (apply snapshot if enabled) ----------
person_in = Person(birthdate=birthdate_in, start_service_db=db_start_in, today=today_in)
early_date_input = add_years(birthdate_in, retire_early_age)

if snapshot_mode:
    # Shift so that 'today' behaves like the early-eligibility date
    delta_days = (today_in - early_date_input).days
    birthdate_eff = shift_date(birthdate_in, delta_days)
    db_start_eff = shift_date(db_start_in, delta_days)
    person = Person(birthdate=birthdate_eff, start_service_db=db_start_eff, today=today_in)

    # Career offset (years from DB start to manager switch) on the real timeline
    mgr_offset_years = max(0.0, years_between(db_start_in, mgr_switch_actual))
    # Effective manager switch on the shifted timeline
    mgr_switch_eff = add_years(db_start_eff, mgr_offset_years)
else:
    person = person_in
    mgr_switch_eff = mgr_switch_actual

# ---------- Timeline & salary projections ----------
age_now = years_between(person.birthdate, person.today)
ages = np.arange(int(math.floor(age_now)), 101)
years_to_65 = int(max(0, math.ceil(retire_normal_age - age_now)))

# Nominal salary growth (forward) from 'today'
nominal_growth_db = (1 + salary_growth_db) * (1 + inflation) - 1
nominal_growth_dc = (1 + salary_growth_dc) * (1 + inflation) - 1

# For DB high-5 approximation: build salary path forward from 'today'
db_salary_series = ann_growth_series(base_salary_now, nominal_growth_db, years_to_65)

# YMPE series and Best Avg YMPE
ympe_series = ann_growth_series(current_ympe, ympe_growth, years_to_65)
def best_avg_ympe(ympes: List[float]) -> float:
    if len(ympes) < 5:
        return float(np.mean(ympes)) if ympes else 0.0
    best = 0.0
    for i in range(0, len(ympes) - 4):
        best = max(best, float(np.mean(ympes[i:i+5])))
    return best
best_avg_ympe_val = best_avg_ympe(ympe_series)

# ---------- DB computation ----------
service_years_to_normal = years_between(person.start_service_db, add_years(person.birthdate, retire_normal_age))
service_years_to_early  = years_between(person.start_service_db, add_years(person.birthdate, retire_early_age))
post2018_start = max(person.start_service_db, dt.date(2018, 1, 1))
post2018_years_early = max(0.0, years_between(post2018_start, add_years(person.birthdate, retire_early_age)))
post2018_years_normal = max(0.0, years_between(post2018_start, add_years(person.birthdate, retire_normal_age)))
db_high5 = db_high5_from_salary_series(db_salary_series)

db_lifetime_early, db_bridge_early = calc_db_lifetime_and_bridge(
    db_high5, best_avg_ympe_val, service_years_to_early, post2018_years_early,
    accrual_rate, integration_rate, bridge_rate
)
db_lifetime_normal, db_bridge_normal = calc_db_lifetime_and_bridge(
    db_high5, best_avg_ympe_val, service_years_to_normal, post2018_years_normal,
    accrual_rate, integration_rate, bridge_rate
)

def db_income_stream(age_start: float, lifetime_amt: float, bridge_amt_to65: float) -> Dict[int, float]:
    incomes = {}
    age_int_start = math.floor(age_start)
    lifetime = lifetime_amt
    bridge = bridge_amt_to65
    for age in range(age_int_start, 101):
        amt = lifetime + (bridge if age < 65 else 0.0)
        incomes[age] = amt
        lifetime *= (1 + index_fraction_cpi * inflation)
        if age < 65:
            bridge *= (1 + index_fraction_cpi * inflation)
    return incomes

db_income_early = db_income_stream(retire_early_age, db_lifetime_early, db_bridge_early)
db_income_normal = db_income_stream(retire_normal_age, db_lifetime_normal, db_bridge_normal)

# ---------- DC contributions using rollback path from manager switch to retirement ----------
dc_total_contrib_rate = employee_dc + employer_match

def build_rolled_back_salary_series(base_today: float, years: int, rollback: float) -> List[float]:
    """
    From 'years' ago to today: salary path that rolls BACK from today's base by 'rollback',
    i.e., S0 = base_today / (1+rollback)^years, then grows forward each year at (1+rollback) to reach base_today.
    """
    if years <= 0:
        return []
    s0 = base_today / ((1 + rollback) ** years)
    return ann_growth_series(s0, rollback, years)

# Determine the contribution window(s)
early_date_eff = add_years(person.birthdate, retire_early_age)
normal_date_eff = add_years(person.birthdate, retire_normal_age)

years_from_switch_to_early  = max(0, int(math.ceil(years_between(mgr_switch_eff, early_date_eff))))
years_from_switch_to_normal = max(0, int(math.ceil(years_between(mgr_switch_eff, normal_date_eff))))

# Build rolled-back base salary paths up to early/normal
rolled_base_to_early  = build_rolled_back_salary_series(base_salary_now, years_from_switch_to_early, rollback_rate)
rolled_base_to_normal = build_rolled_back_salary_series(base_salary_now, years_from_switch_to_normal, rollback_rate)

# Apply manager premium at the switch point (affects whole path thereafter)
rolled_base_to_early  = [s * (1 + manager_premium) for s in rolled_base_to_early]
rolled_base_to_normal = [s * (1 + manager_premium) for s in rolled_base_to_normal]

# Pensionable (STIP up to 50%), optionally also rolled back (we already rolled back base; STIP % applies to that base)
def pensionable_from_base_series(series: List[float], stip_ratio: float) -> List[float]:
    cap = 0.50
    return [s * (1 + min(stip_ratio, cap)) for s in series]

pensionable_to_early  = pensionable_from_base_series(rolled_base_to_early,  stip_share if apply_rollback_to_stip else 0.0)
pensionable_to_normal = pensionable_from_base_series(rolled_base_to_normal, stip_share if apply_rollback_to_stip else 0.0)

# Yearly DC contributions
contribs_to_early  = [pe * dc_total_contrib_rate for pe in pensionable_to_early]
contribs_to_normal = [pe * dc_total_contrib_rate for pe in pensionable_to_normal]

# Accumulate with investment return to get balances at early & normal
dc_balance_at_early  = accumulate_stream(contribs_to_early,  dc_return_net)
dc_balance_at_normal = accumulate_stream(contribs_to_normal, dc_return_net)

# ---------- Side savings from after-tax premium, also rolled back if selected ----------
if apply_rollback_to_side:
    # After-tax premium stream rolls back with the same rollback path
    after_tax_prem_series_early  = [ (b / (1 + manager_premium)) * manager_premium * after_tax_multiplier for b in rolled_base_to_early ]
    after_tax_prem_series_normal = [ (b / (1 + manager_premium)) * manager_premium * after_tax_multiplier for b in rolled_base_to_normal ]
else:
    # Flat based on today's base (less realistic, included for completeness)
    base_after_tax_premium = (base_salary_now * manager_premium) * after_tax_multiplier
    after_tax_prem_series_early  = [base_after_tax_premium]  * years_from_switch_to_early
    after_tax_prem_series_normal = [base_after_tax_premium]  * years_from_switch_to_normal

side_contribs_to_early  = [amt * save_fraction_of_premium for amt in after_tax_prem_series_early]
side_contribs_to_normal = [amt * save_fraction_of_premium for amt in after_tax_prem_series_normal]

side_balance_at_early  = accumulate_stream(side_contribs_to_early,  dc_return_net)
side_balance_at_normal = accumulate_stream(side_contribs_to_normal, dc_return_net)

# ---------- Income streams (DC: 4% rule baseline) ----------
def dc_withdrawal_4pct_stream(start_age: float, start_balance: float, ages_arr: np.ndarray, net_r: float) -> Dict[int, float]:
    incomes = {}
    bal = start_balance
    for age in ages_arr:
        a = int(age)
        if age >= start_age:
            w = bal * 0.04
            incomes[a] = incomes.get(a, 0.0) + w
            bal = (bal - w) * (1 + net_r)
        else:
            incomes[a] = 0.0
    return incomes

def combine_streams(*streams: Dict[int, float]) -> Dict[int, float]:
    res = {}
    for s in streams:
        for k, v in s.items():
            res[k] = res.get(k, 0.0) + v
    return res

def indexed_from_65(base_today: float, infl: float) -> Dict[int, float]:
    res = {}
    amt = base_today
    for age in range(65, 101):
        res[age] = amt
        amt *= (1 + infl)
    return res

cpp_stream = indexed_from_65(cpp_at_65_today, inflation)
oas_stream = indexed_from_65(oas_at_65_today, inflation)

dc_income_early_4pct  = dc_withdrawal_4pct_stream(retire_early_age,  dc_balance_at_early,  ages, dc_return_net)
dc_income_normal_4pct = dc_withdrawal_4pct_stream(retire_normal_age, dc_balance_at_normal, ages, dc_return_net)
side_income_early_4pct  = dc_withdrawal_4pct_stream(retire_early_age,  side_balance_at_early,  ages, dc_return_net)
side_income_normal_4pct = dc_withdrawal_4pct_stream(retire_normal_age, side_balance_at_normal, ages, dc_return_net)

db_total_early  = db_income_early.copy()
db_total_normal = db_income_normal.copy()
dc_total_early_4pct  = combine_streams(dc_income_early_4pct,  side_income_early_4pct)
dc_total_normal_4pct = combine_streams(dc_income_normal_4pct, side_income_normal_4pct)

if show_cpp_oas:
    db_total_early  = combine_streams(db_total_early,  cpp_stream, oas_stream)
    db_total_normal = combine_streams(db_total_normal, cpp_stream, oas_stream)
    dc_total_early_4pct  = combine_streams(dc_total_early_4pct,  cpp_stream, oas_stream)
    dc_total_normal_4pct = combine_streams(dc_total_normal_4pct, cpp_stream, oas_stream)

# ---------- Charts ----------
tab1, tab2, tab3 = st.tabs(["Annual Income (DB vs DC)", "Depletion Test (Match DB Payouts)", "Inputs Snapshot"])

with tab1:
    st.subheader("Annual Income by Age (Nominal $ under selected inflation)")
    if snapshot_mode:
        st.info("Snapshot mode: Today = early-eligibility; DC contributions replayed from the same career offset as your actual manager switch.\n"
                f"Rollback rate for past salaries: {rollback_rate:.2%}")

    def dict_to_series(d: Dict[int, float], ages_list: np.ndarray) -> List[float]:
        return [d.get(int(a), 0.0) for a in ages_list]
    y_db_e = dict_to_series(db_total_early, ages)
    y_db_n = dict_to_series(db_total_normal, ages)
    y_dc_e = dict_to_series(dc_total_early_4pct, ages)
    y_dc_n = dict_to_series(dc_total_normal_4pct, ages)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ages, y=y_db_e, mode="lines", name="DB — retire early"))
    fig.add_trace(go.Scatter(x=ages, y=y_dc_e, mode="lines", name="DC — retire early (4% rule)"))
    fig.add_trace(go.Scatter(x=ages, y=y_db_n, mode="lines", name="DB — retire at 65"))
    fig.add_trace(go.Scatter(x=ages, y=y_dc_n, mode="lines", name="DC — retire at 65 (4% rule)"))
    fig.update_layout(height=520, xaxis_title="Age", yaxis_title="Annual Income ($)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("DC Balance if You Try to Match DB Payouts")
    if snapshot_mode:
        st.info("Includes DC contributions built from rolled-back salaries (and STIP/premium if toggled) from the effective manager switch to retirement.")

    def simulate_matching_db(start_age: float, start_balance: float, payout_dict: Dict[int, float], net_r: float) -> Tuple[np.ndarray, np.ndarray]:
        bal = start_balance
        xs, ys = [], []
        for age in ages:
            xs.append(age); ys.append(bal)
            if age >= start_age:
                w = payout_dict.get(int(age), 0.0)
                bal = (bal - w) * (1 + net_r)
        return np.array(xs), np.array(ys)

    include_side = st.checkbox("Include side savings when matching DB", value=True)
    start_bal_early  = dc_balance_at_early  + (side_balance_at_early  if include_side else 0.0)
    start_bal_normal = dc_balance_at_normal + (side_balance_at_normal if include_side else 0.0)

    x_e, y_e = simulate_matching_db(retire_early_age, start_bal_early, db_income_early, dc_return_net)
    x_n, y_n = simulate_matching_db(retire_normal_age, start_bal_normal, db_income_normal, dc_return_net)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=x_e, y=y_e, mode="lines", name="Match DB — retire early"))
    fig2.add_trace(go.Scatter(x=x_n, y=y_n, mode="lines", name="Match DB — retire at 65"))
    fig2.update_layout(height=520, xaxis_title="Age", yaxis_title="DC Balance ($)")
    st.plotly_chart(fig2, use_container_width=True)

    def depletion_age(x: np.ndarray, y: np.ndarray) -> str:
        below = np.where(y <= 0)[0]
        return f"{int(x[below[0]])}" if len(below) else "Does not deplete by age 100"
    colA, colB = st.columns(2)
    with colA: st.metric("Depletion age (early)", depletion_age(x_e, y_e))
    with colB: st.metric("Depletion age (65)", depletion_age(x_n, y_n))

with tab3:
    st.subheader("Key Inputs Snapshot")
    base_snapshot = {
        "Birthdate (input)": birthdate_in.isoformat(),
        "DB start (input)": db_start_in.isoformat(),
        "Today (input)": today_in.isoformat(),
        "Manager/DC switch (actual)": mgr_switch_actual.isoformat(),
        "Early date (input)": add_years(birthdate_in, retire_early_age).isoformat(),
    }
    if snapshot_mode:
        shifted = {
            "Effective Birthdate (shifted)": person.birthdate.isoformat(),
            "Effective DB Start (shifted)": person.start_service_db.isoformat(),
            "Effective Manager Switch (shifted)": add_years(person.start_service_db, years_between(db_start_in, mgr_switch_actual)).isoformat(),
            "Note": "Snapshot mode with rollback: DC built from rolled-back salaries then grown at investment return.",
            "Rollback rate": f"{rollback_rate:.2%}",
        }
        base_snapshot.update(shifted)

    snapshot = {
        **base_snapshot,
        "Inflation": f"{inflation:.2%}",
        "YMPE growth": f"{ympe_growth:.2%}",
        "YMPE now": f"${current_ympe:,.0f}",
        "Base salary (DB today)": f"${base_salary_now:,.0f}",
        "Manager premium": f"{manager_premium:.0%}",
        "Salary growth DB (real)": f"{salary_growth_db:.2%}",
        "Salary growth DC (real)": f"{salary_growth_dc:.2%}",
        "STIP % of base": f"{stip_share:.0%} (cap 50%)",
        "DC employee %": f"{employee_dc:.2%}",
        "DC employer %": f"{employer_match:.2%}",
        "DC nominal return": f"{dc_return_nominal:.2%}",
        "Fee drag": f"{fee_drag:.2%}",
        "DC net return": f"{dc_return_net:.2%}",
        "After-tax premium multiplier": f"{after_tax_multiplier:.0%}",
        "Save % of after-tax premium": f"{save_fraction_of_premium:.0%}",
        "Side savings nominal return": f"{side_return_nominal:.2%}",
        "Retire early age": f"{retire_early_age}",
        "Retire normal age": f"{retire_normal_age}",
        "CPP @65 (today $)": f"${cpp_at_65_today:,.0f}",
        "OAS @65 (today $)": f"${oas_at_65_today:,.0f}",
        "DB accrual": f"{accrual_rate:.3%}",
        "DB integration": f"{integration_rate:.3%}",
        "DB bridge": f"{bridge_rate:.3%}",
        "DB indexing (% of CPI)": f"{index_fraction_cpi:.0%}",
    }
    st.dataframe(pd.DataFrame.from_dict(snapshot, orient="index", columns=["Value"]))
