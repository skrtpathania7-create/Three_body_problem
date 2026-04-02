import numpy as np
import pandas as pd

# Physics Constants
G = 1.0
dt = 0.05
max_steps = 1000
data_rows = []

def get_acceleration(pos1, pos2, m2):
    r_vec = pos2 - pos1
    dist = np.linalg.norm(r_vec) + 0.1 # "Softening" to prevent infinite force
    return G * m2 * r_vec / (dist**3)

print("Generating Physics Data...")

for i in range(100): # Number of simulations
    # Random initial conditions for 3 bodies
    # Mass, X, Y, VX, VY
    m = np.array([10.0, 10.0, 10.0])
    pos = np.random.uniform(-10, 10, (3, 2))
    vel = np.random.uniform(-0.5, 0.5, (3, 2))
    
    # Store starting state for the AI
    start_state = np.hstack([pos.flatten(), vel.flatten()])
    
    stable = 1
    for step in range(max_steps):
        # Calculate total force on each body
        accel = np.zeros((3, 2))
        accel[0] = get_acceleration(pos[0], pos[1], m[1]) + get_acceleration(pos[0], pos[2], m[2])
        accel[1] = get_acceleration(pos[1], pos[0], m[0]) + get_acceleration(pos[1], pos[2], m[2])
        accel[2] = get_acceleration(pos[2], pos[0], m[0]) + get_acceleration(pos[2], pos[1], m[1])
        
        vel += accel * dt
        pos += vel * dt
        
        # Stability Check: If any planet flies too far, it's unstable
        if np.any(np.abs(pos) > 50):
            stable = 0
            break
            
    data_rows.append(np.append(start_state, stable))

# Save to CSV
columns = ['x1','y1','x2','y2','x3','y3','vx1','vy1','vx2','vy2','vx3','vy3','stable']
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv('orbital_data.csv', index=False)
print("Saved 100 simulations to orbital_data.csv")






