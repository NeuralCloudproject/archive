import numpy as np
from scipy import integrate

# 경로 정의
def gamma(s):
    return np.array([s, s])

# 함수 정의
def f_regular(x):
    return x[0] * x[1]

# 경로 길이 계산
def arc_length(a, b, N=1000):
    s = np.linspace(a, b, N)
    dg_ds = np.ones_like(s) * np.sqrt(2)
    return integrate.simpson(dg_ds, s)

# UAI 시뮬레이션
def uai_simulation(f, gamma, a, b, samples=100000):
    L_gamma = arc_length(a, b)
    s_samples = np.random.uniform(a, b, samples)
    gamma_samples = np.array([gamma(s) for s in s_samples])
    f_values = np.array([f(g) for g in gamma_samples])
    integral = np.mean(f_values) * (b - a)
    integral_err = np.std(f_values) / np.sqrt(samples) * (b - a)
    result = L_gamma * integral
    return result, L_gamma * integral_err

# 실행
a, b = -1, 1
paper_result = 4 * np.sqrt(2) / 3
sim_result, sim_err = uai_simulation(f_regular, gamma, a, b)

print(f"Paper Prediction: {paper_result:.6f}")
print(f"Simulation Result: {sim_result:.6f} ± {sim_err:.6f}")
print(f"Absolute Error: {abs(paper_result - sim_result):.6f}")
print(f"Relative Error: {abs(paper_result - sim_result) / paper_result * 100:.2f}%")

print("---------------------------------------------------------------------------")

# 경로 정의
def gamma(s):
    return np.array([s])

# 함수 정의
def f_distribution(x):
    s = x[0]
    return 1 / (1 + np.abs(s))  # f * mu 직접 계산

# 경로 길이 계산 (참고용, 여기서는 사용 안 함)
def arc_length(a, b, N=1000):
    s = np.linspace(a, b, N)
    dg_ds = np.ones_like(s)
    L = integrate.simpson(dg_ds, s)
    print(f"Debug: L_gamma = {L:.6f}")
    return L

# UAI 시뮬레이션
def uai_simulation(f, gamma, a, b, samples=100000):
    s_samples = np.random.uniform(a, b, samples)
    mask = np.abs(s_samples) > 1e-6  # 특이점 제외
    s_samples = s_samples[mask]
    gamma_samples = np.array([gamma(s) for s in s_samples])
    f_values = np.array([f(g) for g in gamma_samples])
    integral = np.mean(f_values) * (b - a)
    integral_err = np.std(f_values) / np.sqrt(len(s_samples)) * (b - a)
    
    print(f"Debug: Mean f_values = {np.mean(f_values):.6f}")
    print(f"Debug: Integral = {integral:.6f}")
    
    return integral, integral_err  # L_gamma 곱하지 않음

# 실행
a, b = -1, 1
paper_result = 2 * np.log(2)
sim_result, sim_err = uai_simulation(f_distribution, gamma, a, b)

print(f"Paper Prediction: {paper_result:.6f}")
print(f"Simulation Result: {sim_result:.6f} ± {sim_err:.6f}")
print(f"Absolute Error: {abs(paper_result - sim_result):.6f}")
print(f"Relative Error: {abs(paper_result - sim_result) / paper_result * 100:.2f}%")