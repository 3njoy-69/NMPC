import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import json

# --- THÔNG SỐ MÔ HÌNH ---
L = 0.43 * 100  # chiều dài cơ sở (mm)
dt = 0.1        # thời gian mẫu (s)
N = 20          # horizon

# --- ĐẶT BÀI TOÁN CASADI ---
opti = ca.Opti()

X = opti.variable(3, N + 1)   # trạng thái: x, y, yaw
U = opti.variable(2, N)       # điều khiển: v, delta
x0 = opti.parameter(3)        # trạng thái ban đầu
xref = opti.parameter(3, N + 1)  # tham chiếu

# --- HÀM ĐỘNG HỌC ---
def bicycle_model(x, u):
    theta = x[2]
    v = u[0]
    delta = u[1]
    return ca.vertcat(
        v * ca.cos(theta) * 100,
        v * ca.sin(theta) * 100,
        v * 100 / L * ca.tan(delta)
    )

# --- RÀNG BUỘC ĐỘNG HỌC ---
for k in range(N):
    x_next = X[:, k] + dt * bicycle_model(X[:, k], U[:, k])
    opti.subject_to(X[:, k + 1] == x_next)

opti.subject_to(X[:, 0] == x0)  # ràng buộc trạng thái ban đầu
opti.subject_to(opti.bounded(-1.0, U[0, :], 1.0))  # giới hạn tốc độ
opti.subject_to(opti.bounded(-np.deg2rad(30), U[1, :], np.deg2rad(30)))  # giới hạn góc lái

# --- RÀNG BUỘC DỪNG XE Ở CUỐI ---
opti.subject_to(U[0, -1] == 0)  # ép vận tốc v = 0 tại bước cuối cùng

# --- HÀM MỤC TIÊU ---
Q = np.diag([10, 10, 0.5])  # trọng số trạng thái
R = np.diag([0.1, 50])      # trọng số điều khiển
cost = 0

for k in range(N):
    dx = X[:, k] - xref[:, k]
    cost += ca.mtimes([dx.T, Q, dx]) + ca.mtimes([U[:, k].T, R, U[:, k]])
dxN = X[:, N] - xref[:, N]
cost += ca.mtimes([dxN.T, Q, dxN])

opti.minimize(cost)

# --- SETUP SOLVER ---
opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

# --- ĐỌC DỮ LIỆU TỪ FILE JSON ---
with open("path_data.json", "r") as f:
    ref_list = json.load(f)
ref_array = np.array(ref_list).T
xref_traj = ref_array

T = ref_array.shape[1]  # chạy hết các node
assert T > 0, "Dữ liệu quỹ đạo không hợp lệ."

# --- SIMULATE ---
x_init = ref_array[:, 0]
x_log = [x_init]
u_log = []
x_current = x_init

for t in range(T):
    opti.set_value(x0, x_current)

    # Xử lý trường hợp t+N > T
    if t + N + 1 <= T:
        xref_window = xref_traj[:, t:t+N+1]
    else:
        last_col = np.tile(xref_traj[:, -1:], (1, t + N + 1 - T))
        xref_window = np.concatenate([xref_traj[:, t:], last_col], axis=1)

    opti.set_value(xref, xref_window)

    try:
        sol = opti.solve()
        u_opt = sol.value(U[:, 0])
    except RuntimeError:
        print(f"[\u274c] MPC lỗi tại bước {t}, dùng u = 0.")
        u_opt = np.array([0.0, 0.0])

    print(f"Bước {t:03d}: v = {u_opt[0]:.3f} m/s, delta = {np.rad2deg(u_opt[1]):.2f}°")
    print(f"  x_ref = {xref_traj[:, t]}")

    u_log.append(u_opt)
    x_next = x_current + dt * np.array(bicycle_model(x_current, u_opt)).flatten()
    x_log.append(x_next)
    x_current = x_next

x_log = np.array(x_log)
u_log = np.array(u_log)

# --- VẼ ĐỒ THỊ ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(xref_traj[0, :], xref_traj[1, :], 'g--', label='Quỹ đạo tham chiếu')
plt.plot(x_log[:, 0], x_log[:, 1], 'b-', label='Quỹ đạo thực tế')
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.step(range(len(u_log)), u_log[:, 0], label='v (m/s)')
plt.step(range(len(u_log)), np.rad2deg(u_log[:, 1]), label='delta (deg)')
plt.xlabel("Thời gian bước (k)")
plt.ylabel("Điều khiển")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
