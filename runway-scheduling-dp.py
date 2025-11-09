import random
import math
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# === Page Config ===
st.set_page_config(
    page_title="✈️ Aircraft Landing Scheduler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Separation Function ===
def get_sep(leader_type, follower_type):
    sep_matrix = {
        'H': {'H': 4, 'UM': 5, 'LM': 5, 'S': 5},
        'UM': {'H': 6, 'UM': 6, 'LM': 5, 'S': 4},
        'LM': {'H': 5, 'UM': 5, 'LM': 5, 'S': 5},
        'S': {'H': 3, 'UM': 3, 'LM': 3, 'S': 3}
    }
    return sep_matrix.get(leader_type, {}).get(follower_type, 3)

# === GA Evaluation ===
def evaluate(candidate_times, aircraft):
    P = len(aircraft)
    order = sorted(range(P), key=lambda i: candidate_times[i])
    violations = sum(max(0, aircraft[i][2] - candidate_times[i]) for i in range(P))
    for k in range(1, P):
        prev_i = order[k-1]
        curr_i = order[k]
        delta = candidate_times[curr_i] - candidate_times[prev_i]
        violations += max(0, get_sep(aircraft[prev_i][0], aircraft[curr_i][0]) - delta)
    fitness = sum(max(0, candidate_times[i] - aircraft[i][1])**2 for i in range(P))
    return fitness, violations

def is_better(eval1, eval2):
    f1, u1 = eval1
    f2, u2 = eval2
    if u1 != u2:
        return u1 < u2
    return f1 < f2

# === GA Helpers ===
def tournament_select(pop, aircraft):
    a, b = random.sample(pop, 2)
    return a if is_better(evaluate(a, aircraft), evaluate(b, aircraft)) else b

def uniform_crossover(p1, p2):
    return [random.choice([p1[i], p2[i]]) for i in range(len(p1))]

def find_worst(pop, aircraft):
    worst_idx = 0
    worst_eval = evaluate(pop[0], aircraft)
    for i in range(1, len(pop)):
        curr_eval = evaluate(pop[i], aircraft)
        if not is_better(curr_eval, worst_eval):
            worst_idx, worst_eval = i, curr_eval
    return worst_idx, worst_eval

def init_population(size, aircraft):
    P = len(aircraft)
    return [[aircraft[i][1] + random.uniform(-200, 1200) for i in range(P)] for _ in range(size)]

def run_ga(aircraft, pop_size=50, generations=1000, progress_callback=None):
    population = init_population(pop_size, aircraft)
    best_scores, best_violations = [], []

    for gen in range(generations):
        p1 = tournament_select(population, aircraft)
        p2 = tournament_select(population, aircraft)
        child = uniform_crossover(p1, p2)
        child_eval = evaluate(child, aircraft)
        worst_idx, worst_eval = find_worst(population, aircraft)
        if is_better(child_eval, worst_eval):
            population[worst_idx] = child
        if gen % 50 == 0:
            best_idx = min(range(len(population)), key=lambda i: evaluate(population[i], aircraft))
            best_eval = evaluate(population[best_idx], aircraft)
            best_scores.append(best_eval[0])
            best_violations.append(best_eval[1])
            if progress_callback:
                progress_callback(gen, best_scores, best_violations)

    best_idx = min(range(len(population)), key=lambda i: evaluate(population[i], aircraft))
    return population[best_idx], best_scores, best_violations

# === Sidebar Options ===
st.sidebar.title("⚙️ Scheduler Settings")
num_aircraft = st.sidebar.slider("Number of Aircraft", min_value=10, max_value=100, value=20, step=1)
pop_size = st.sidebar.number_input("Population Size", min_value=10, max_value=200, value=50)
generations = st.sidebar.number_input("Generations", min_value=100, max_value=5000, value=1000)

# === Load or Generate Aircraft Data ===
if "aircraft_data" not in st.session_state:
    # Aircraft types
    aircraft_types = ['H', 'UM', 'LM', 'S']
    generated_aircraft = []
    base_target_time = 1000

    for i in range(num_aircraft):
        aircraft_type = aircraft_types[i % len(aircraft_types)]
        target_time = base_target_time + i * 60
        earliest_time = target_time - 180
        generated_aircraft.append((aircraft_type, target_time, earliest_time))

    st.session_state.aircraft_data = generated_aircraft

# === Aircraft Table Preview ===
st.subheader("🛬 Aircraft Data Preview")
df = pd.DataFrame(
    st.session_state.aircraft_data,
    columns=["Type", "Target Time (s)", "Earliest Time (s)"]
)
st.dataframe(df)

# === Run GA Button ===
if st.button("🚀 Start Scheduling GA"):
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_callback(gen, f_vals, u_vals):
        progress_bar.progress(min(1.0, gen / generations))
        status_text.text(f"Generation {gen}: Fitness={f_vals[-1]:.2f}, Violations={u_vals[-1]:.2f}")

    best_solution, best_scores, best_violations = run_ga(
        st.session_state.aircraft_data,
        pop_size=pop_size,
        generations=generations,
        progress_callback=progress_callback
    )

    st.success("✅ GA Completed!")

    # === Build Landing Schedule Table ===
    order = sorted(range(len(best_solution)), key=lambda i: best_solution[i])
    schedule = []
    total_delay = 0
    for idx in order:
        y_i = best_solution[idx]
        t_i, l_i = st.session_state.aircraft_data[idx][1], st.session_state.aircraft_data[idx][2]
        delay = max(0, y_i - t_i)
        total_delay += delay
        schedule.append([idx, st.session_state.aircraft_data[idx][0], round(y_i, 1), t_i, l_i, round(delay, 1)])

    st.subheader(f"🗂️ Landing Schedule (Total Linear Delay: {round(total_delay, 1)} s)")
    st.table(pd.DataFrame(schedule, columns=["Index", "Type", "Proposed", "Target", "Earliest", "Delay"]))

    # === Plot GA Evolution ===
    st.subheader("📈 GA Evolution Over Generations")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(0, generations, 50), best_scores, label="Fitness (F)", color="cyan", marker='o')
    ax.plot(range(0, generations, 50), best_violations, label="Violations (U)", color="orange", marker='x')
    ax.set_xlabel("Generations")
    ax.set_ylabel("Fitness / Violations")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
