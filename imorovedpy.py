import numpy as np
import pandas as pd

# Physics Constants
G = 1.0
dt = 0.05
max_steps = 1000
NUM_SIMULATIONS = 5000  # Was 100 — more data = smarter model

def get_acceleration(pos1, pos2, m2):
    r_vec = pos2 - pos1
    dist = np.linalg.norm(r_vec) + 0.1  # Softening to prevent infinite force
    return G * m2 * r_vec / (dist ** 3)

print(f"Generating {NUM_SIMULATIONS} physics simulations...")
print("This takes ~2 minutes. Grab a coffee.")

data_rows = []
stable_count = 0
unstable_count = 0

for i in range(NUM_SIMULATIONS):

    # Progress indicator every 500 sims
    if (i + 1) % 500 == 0:
        print(f"  [{i+1}/{NUM_SIMULATIONS}] stable so far: {stable_count}, unstable: {unstable_count}")

    m = np.array([10.0, 10.0, 10.0])
    pos = np.random.uniform(-10, 10, (3, 2))
    vel = np.random.uniform(-0.5, 0.5, (3, 2))

    start_state = np.hstack([pos.flatten(), vel.flatten()])

    stable = 1
    for step in range(max_steps):
        accel = np.zeros((3, 2))
        accel[0] = get_acceleration(pos[0], pos[1], m[1]) + get_acceleration(pos[0], pos[2], m[2])
        accel[1] = get_acceleration(pos[1], pos[0], m[0]) + get_acceleration(pos[1], pos[2], m[2])
        accel[2] = get_acceleration(pos[2], pos[0], m[0]) + get_acceleration(pos[2], pos[1], m[1])

        vel += accel * dt
        pos += vel * dt

        if np.any(np.abs(pos) > 50):
            stable = 0
            break

    if stable == 1:
        stable_count += 1
    else:
        unstable_count += 1

    data_rows.append(np.append(start_state, stable))

# Save to CSV
columns = ['x1','y1','x2','y2','x3','y3','vx1','vy1','vx2','vy2','vx3','vy3','stable']
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv('orbital_data.csv', index=False)

# Class balance report
total = stable_count + unstable_count
print(f"\nDone! Saved {NUM_SIMULATIONS} simulations to orbital_data.csv")
print(f"  Stable:   {stable_count} ({100*stable_count/total:.1f}%)")
print(f"  Unstable: {unstable_count} ({100*unstable_count/total:.1f}%)")

if stable_count / total < 0.3 or stable_count / total > 0.7:
    print("\nWARNING: Class imbalance detected.")
    print("mlparthack.py will handle this automatically with class_weight='balanced'.")
else:
    print("\nClass balance looks good.")










