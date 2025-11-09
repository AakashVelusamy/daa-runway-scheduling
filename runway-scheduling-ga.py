import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Optimal Aircraft Landing Scheduler (DP)",
    layout="wide"
)

st.title("✈️ Optimal Aircraft Landing Scheduler Using Dynamic Programming (DP)")
st.markdown(
    """
This tool calculates the optimal landing times for multiple aircraft considering type-dependent separation constraints and target times.  
All numeric values are displayed without commas for clarity.
"""
)

# === Sidebar Inputs ===
st.sidebar.header("Configuration")
num_aircraft = st.sidebar.slider("Number of Aircraft", min_value=5, max_value=30, value=15, step=1)
base_time = st.sidebar.number_input("Base Target Time (s)", min_value=0, value=1000, step=10)
time_interval = st.sidebar.number_input("Interval Between Aircraft (s)", min_value=10, value=60, step=10)

# Aircraft Types and Separation Matrix
aircraft_types = ['H', 'UM', 'LM', 'S']
sep_dict = {
    'H': {'H': 4, 'UM': 5, 'LM': 5, 'S': 5},
    'UM': {'H': 6, 'UM': 6, 'LM': 5, 'S': 4},
    'LM': {'H': 5, 'UM': 5, 'LM': 5, 'S': 5},
    'S': {'H': 3, 'UM': 3, 'LM': 3, 'S': 3}
}

def get_sep(leader_type, follower_type):
    return sep_dict.get(leader_type, {}).get(follower_type, 3)

# === Generate Aircraft Data Dynamically ===
if "aircraft_data" not in st.session_state:
    aircraft_data = []
    for i in range(num_aircraft):
        t_type = aircraft_types[i % len(aircraft_types)]
        target = base_time + i * time_interval
        earliest = target - 180
        aircraft_data.append((t_type, target, earliest))
    st.session_state.aircraft_data = aircraft_data

df_aircraft = pd.DataFrame(st.session_state.aircraft_data, columns=["Type", "Target Time (s)", "Earliest Time (s)"])
st.subheader("🛬 Aircraft Data Preview")
numeric_cols = ["Target Time (s)", "Earliest Time (s)"]
st.dataframe(df_aircraft.style.format({col: "{:.0f}" for col in numeric_cols}))

# === Run DP Scheduler ===
if st.button("🚀 Calculate Optimal Landing Sequence"):
    st.info("Calculating optimal landing times...")
    
    P = len(st.session_state.aircraft_data)
    INF = 1e18
    
    # Initialize DP tables
    dp_cost = np.full((1 << P, P), INF)
    dp_time = np.zeros((1 << P, P))
    pred = np.full((1 << P, P), -1)
    
    # Initialize single aircraft
    for i in range(P):
        mask = 1 << i
        dp_time[mask][i] = st.session_state.aircraft_data[i][2]  # Earliest
        dp_cost[mask][i] = 0
    
    # Fill DP table
    for mask in range(1, 1 << P):
        for last in range(P):
            if not (mask & (1 << last)):
                continue
            prev_mask = mask ^ (1 << last)
            if prev_mask == 0:
                continue
            min_c, best_t, best_prev = INF, 0, -1
            for prev in range(P):
                if not (prev_mask & (1 << prev)):
                    continue
                t_prev = dp_time[prev_mask][prev]
                sep = get_sep(st.session_state.aircraft_data[prev][0],
                              st.session_state.aircraft_data[last][0])
                t_cand = max(st.session_state.aircraft_data[last][2], t_prev + sep)
                c_add = max(0.0, t_cand - st.session_state.aircraft_data[last][1]) ** 2
                total_c = dp_cost[prev_mask][prev] + c_add
                if total_c < min_c:
                    min_c, best_t, best_prev = total_c, t_cand, prev
            if best_prev != -1:
                dp_cost[mask][last] = min_c
                dp_time[mask][last] = best_t
                pred[mask][last] = best_prev
    
    # Backtrack best sequence
    full_mask = (1 << P) - 1
    best_last = np.argmin(dp_cost[full_mask])
    seq, mask, last = [], full_mask, best_last
    while mask != 0:
        seq.append(last)
        prev = pred[mask][last]
        if prev == -1:
            break
        mask ^= (1 << last)
        last = prev
    seq.reverse()
    
    # Compute landing times
    landing_times = np.zeros(P)
    for k, i in enumerate(seq):
        if k == 0:
            landing_times[i] = st.session_state.aircraft_data[i][2]
        else:
            prev_i = seq[k-1]
            landing_times[i] = max(st.session_state.aircraft_data[i][2],
                                   landing_times[prev_i] + get_sep(st.session_state.aircraft_data[prev_i][0],
                                                                   st.session_state.aircraft_data[i][0]))
    
    total_delay = sum(max(0, landing_times[i] - st.session_state.aircraft_data[i][1]) for i in range(P))
    min_f = sum(max(0, landing_times[i] - st.session_state.aircraft_data[i][1]) ** 2 for i in range(P))
    
    # === Display Results ===
    st.success("✅ Optimal Landing Sequence Calculated!")
    st.subheader(f"Optimal Landing Schedule (Total Linear Delay: {total_delay:.1f} s)")
    
    schedule = []
    for idx in seq:
        t_i = landing_times[idx]
        t_target = st.session_state.aircraft_data[idx][1]
        t_earliest = st.session_state.aircraft_data[idx][2]
        delay = max(0, t_i - t_target)
        schedule.append([idx, st.session_state.aircraft_data[idx][0],
                         round(t_i,1), t_target, t_earliest, round(delay,1)])
    
    st.table(pd.DataFrame(schedule, columns=["Index", "Type", "Landing Time (s)", "Target (s)", "Earliest (s)", "Delay (s)"]))
    
    # === Plot Landing Sequence ===
    st.subheader("Landing Sequence Timeline")
    fig, ax = plt.subplots(figsize=(12,4))
    colors = {"H":"#1f77b4", "UM":"#ff7f0e", "LM":"#2ca02c", "S":"#d62728"}
    for idx in seq:
        ax.barh(f"Aircraft {idx} ({st.session_state.aircraft_data[idx][0]})",
                landing_times[idx],
                color=colors[st.session_state.aircraft_data[idx][0]])
    ax.set_xlabel("Landing Time (s)")
    ax.set_ylabel("Aircraft")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
