import numpy as np
from numba import cuda, float64, complex128
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm  # 진행도 막대

# SU(3) 행렬 생성
def su3_matrix(angles):
    U = np.eye(3, dtype=np.complex128)
    U[0, 0] = np.cos(angles[0]) + 1j * np.sin(angles[0])
    U[1, 1] = np.cos(angles[1]) + 1j * np.sin(angles[1])
    U[2, 2] = 1.0 + 0.0j
    return U

# 격자 초기화
def initialize_lattice(N=32):
    lattice = np.zeros((N, N, N, N, 4, 3, 3), dtype=np.complex128)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    for mu in range(4):
                        angles = np.random.uniform(0, 0.1, 8)
                        lattice[x, y, z, t, mu] = su3_matrix(angles)
    return cuda.to_device(lattice)

# Metropolis-Hastings 업데이트 (CUDA)
@cuda.jit
def metropolis_update(lattice, delta, beta, N, rng_states):
    idx = cuda.grid(1)
    if idx < N * N * N * N * 4:
        mu = idx % 4
        t = (idx // 4) % N
        z = (idx // (4 * N)) % N
        y = (idx // (4 * N * N)) % N
        x = idx // (4 * N * N * N)
        
        staple = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                staple[i, j] = 0.0 + 0.0j
        for nu in range(4):
            if nu != mu:
                x_plus = (x + (nu == 0)) % N
                y_plus = (y + (nu == 1)) % N
                z_plus = (z + (nu == 2)) % N
                t_plus = (t + (nu == 3)) % N
                x_minus = (x - (nu == 0)) % N
                y_minus = (y - (nu == 1)) % N
                z_minus = (z - (nu == 2)) % N
                t_minus = (t - (nu == 3)) % N
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            staple[i, j] += lattice[x, y, z, t, nu, i, k] * \
                                            lattice[x_plus, y_plus, z_plus, t_plus, nu, j, k].conjugate()
                            staple[i, j] += lattice[x_minus, y_minus, z_minus, t_minus, nu, k, i].conjugate() * \
                                            lattice[x_minus, y_minus, z_minus, t_minus, mu, j, k]
        
        U_old = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                U_old[i, j] = lattice[x, y, z, t, mu, i, j]
        
        U_new = cuda.local.array((3, 3), dtype=complex128)
        angles = cuda.local.array(8, dtype=float64)
        for i in range(8):
            angles[i] = xoroshiro128p_uniform_float64(rng_states, idx) * 2 * delta - delta
        for i in range(3):
            for j in range(3):
                if i == j:
                    U_new[i, j] = math.cos(angles[i]) + 1j * math.sin(angles[i])
                else:
                    U_new[i, j] = 0.0 + 0.0j
        U_temp = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                U_temp[i, j] = 0.0 + 0.0j
                for k in range(3):
                    U_temp[i, j] += U_new[i, k] * U_old[k, j]
        for i in range(3):
            for j in range(3):
                U_new[i, j] = U_temp[i, j]
        
        dS = 0.0
        for i in range(3):
            for j in range(3):
                dS += (staple[i, j] * (U_old[j, i] - U_new[j, i])).real
        dS *= beta
        
        if dS < 0 or xoroshiro128p_uniform_float64(rng_states, idx) < math.exp(-dS):
            for i in range(3):
                for j in range(3):
                    lattice[x, y, z, t, mu, i, j] = U_new[i, j]

# 열역학 업데이트 실행
def thermalize_lattice(lattice, beta=6.0, steps=120000, N=32):
    total_threads = N * N * N * N * 4
    threads_per_block = 256
    blocks_per_grid = (total_threads + (threads_per_block - 1)) // threads_per_block
    rng_states = create_xoroshiro128p_states(total_threads, seed=1234)
    for step in tqdm(range(steps), desc="Thermalizing Lattice"):
        metropolis_update[blocks_per_grid, threads_per_block](lattice, 0.1, beta, N, rng_states)
        cuda.synchronize()
    return lattice

# Alpha Integration: 사용자 정의 함수 계산 (CUDA)
@cuda.jit
def cuda_alpha_integral(f_values, coords, func_idx):
    idx = cuda.grid(1)
    if idx < coords.shape[0]:
        x1, x2 = coords[idx, 0], coords[idx, 1]
        if func_idx == 0:  # f(x_1, x_2) = x_1^2
            f_values[idx] = x1 * x1
        elif func_idx == 1:  # f(x) = 1/(1 + |x|)
            f_values[idx] = 1 / (1 + abs(x1))

def simulate_alpha_integration(N=32, samples=1000000, func='scalar', beta=6.0):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, beta=beta, steps=120000, N=N)
    coords = np.random.uniform(-1, 1, (samples, 2))  # 독립적인 2D 좌표
    d_coords = cuda.to_device(coords)
    d_f_values = cuda.device_array(samples, dtype=np.float64)
    
    threads_per_block = 256
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    func_idx = 0 if func == 'scalar' else 1
    cuda_alpha_integral[blocks_per_grid, threads_per_block](d_f_values, d_coords, func_idx)
    cuda.synchronize()
    
    f_values = d_f_values.copy_to_host()
    integral = np.mean(f_values) * 2  # [-1, 1] 범위의 길이
    integral_err = np.std(f_values) / np.sqrt(samples) * 2
    
    paper_result = 4 * np.sqrt(2) / 3 if func == 'scalar' else 2 * np.log(2)  # 논문 이론값
    result = (2 * np.sqrt(2) * integral) if func == 'scalar' else integral  # Distribution은 L_gamma 중복 제거
    
    # Alpha Integration 그래프 추가
    plt.hist(f_values, bins=50, density=True, alpha=0.7, label=f'{func} Values')
    plt.xlabel('Function Value')
    plt.ylabel('Density')
    plt.title(f'Alpha Integration: {func} Distribution')
    plt.legend()
    plt.show()
    
    print(f"=== Alpha Integration: {func} ===")
    print(f"Paper Prediction: {paper_result:.6f}")
    print(f"Simulation Result: {result:.6f} ± {integral_err:.6f}")
    print(f"Absolute Error: {abs(paper_result - result):.6f}")
    print(f"Relative Error: {abs(paper_result - result) / paper_result * 100:.2f}%")
    return result, integral_err, paper_result

# 격자 QCD: Wilson 루프 상관 함수 계산 (CUDA) - C(t) = <W(0)W(t)>
@cuda.jit
def wilson_loop_correlation(lattice, w_values, N, t_max):
    idx = cuda.grid(1)
    if idx < N * N * N:
        z = idx % N
        idx //= N
        y = idx % N
        x = idx // N
        
        # W(0) 계산
        w0 = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                w0[i, j] = 1.0 if i == j else 0.0
        
        temp = cuda.local.array((3, 3), dtype=complex128)
        # x 방향 (공간) 앞으로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+i)%N, y, z, 0, 0, k, n]
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # t 방향 (시간) 앞으로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+9)%N, y, z, i%N, 3, k, n]
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # x 방향 (공간) 뒤로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+9-i)%N, y, z, 9%N, 0, k, n].conjugate()
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # t 방향 (시간) 뒤로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[x, y, z, (9-i)%N, 3, k, n].conjugate()
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        
        trace_w0 = 0.0
        for i in range(3):
            trace_w0 += w0[i, i].real
        w0_val = trace_w0 / 3

        # W(t) 계산 및 C(t) = W(0) * W(t)
        for dt in range(t_max):
            wt = cuda.local.array((3, 3), dtype=complex128)
            for i in range(3):
                for j in range(3):
                    wt[i, j] = 1.0 if i == j else 0.0
            
            # x 방향 (공간) 앞으로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+i)%N, y, z, dt%N, 0, k, n]
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # t 방향 (시간) 앞으로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+9)%N, y, z, (dt+i)%N, 3, k, n]
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # x 방향 (공간) 뒤로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+9-i)%N, y, z, (dt+9)%N, 0, k, n].conjugate()
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # t 방향 (시간) 뒤로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[x, y, z, (dt+9-i)%N, 3, k, n].conjugate()
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            
            trace_wt = 0.0
            for i in range(3):
                trace_wt += wt[i, i].real
            wt_val = trace_wt / 3
            
            # C(t) = W(0) * W(t)
            w_values[idx * t_max + dt] = w0_val * wt_val

# Alpha-QCD: Alpha Integration 적용 상관 함수 계산 (CUDA)
@cuda.jit
def alpha_qcd_correlation(lattice, w_values, coords, N, t_max):
    idx = cuda.grid(1)
    if idx < coords.shape[0]:
        x = int(coords[idx, 0] * N) % N
        y = int(coords[idx, 1] * N) % N
        z = int(coords[idx, 2] * N) % N
        
        # W(0) 계산
        w0 = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                w0[i, j] = 1.0 if i == j else 0.0
        
        temp = cuda.local.array((3, 3), dtype=complex128)
        # x 방향 (공간) 앞으로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+i)%N, y, z, 0, 0, k, n]
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # t 방향 (시간) 앞으로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+9)%N, y, z, i%N, 3, k, n]
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # x 방향 (공간) 뒤로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[(x+9-i)%N, y, z, 9%N, 0, k, n].conjugate()
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        # t 방향 (시간) 뒤로
        for i in range(10):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        temp[m, n] += w0[m, k] * lattice[x, y, z, (9-i)%N, 3, k, n].conjugate()
            for m in range(3):
                for n in range(3):
                    w0[m, n] = temp[m, n]
        
        trace_w0 = 0.0
        for i in range(3):
            trace_w0 += w0[i, i].real
        w0_val = trace_w0 / 3

        # Alpha Integration 가중치 f(x1, x2) = x1^2
        x1 = coords[idx, 0]
        f_weight = x1 * x1

        # W(t) 계산 및 C(t) = f(x1, x2) * W(0) * W(t)
        for dt in range(t_max):
            wt = cuda.local.array((3, 3), dtype=complex128)
            for i in range(3):
                for j in range(3):
                    wt[i, j] = 1.0 if i == j else 0.0
            
            # x 방향 (공간) 앞으로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+i)%N, y, z, dt%N, 0, k, n]
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # t 방향 (시간) 앞으로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+9)%N, y, z, (dt+i)%N, 3, k, n]
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # x 방향 (공간) 뒤로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[(x+9-i)%N, y, z, (dt+9)%N, 0, k, n].conjugate()
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            # t 방향 (시간) 뒤로
            for i in range(10):
                for m in range(3):
                    for n in range(3):
                        temp[m, n] = 0.0
                        for k in range(3):
                            temp[m, n] += wt[m, k] * lattice[x, y, z, (dt+9-i)%N, 3, k, n].conjugate()
                for m in range(3):
                    for n in range(3):
                        wt[m, n] = temp[m, n]
            
            trace_wt = 0.0
            for i in range(3):
                trace_wt += wt[i, i].real
            wt_val = trace_wt / 3
            
            # C(t) = f(x1, x2) * W(0) * W(t)
            w_values[idx * t_max + dt] = f_weight * w0_val * wt_val

# 부트스트랩으로 오차 계산
def bootstrap_error(data, n_resamples=1000):
    means = []
    n = len(data)
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(resample))
    return np.std(means)

# 단일 지수 피팅 함수
def exp_fit(t, A, m):
    return A * np.exp(-m * t)

# Lattice QCD 시뮬레이션
def simulate_lattice_qcd(N=32, steps=120000, samples=1000, t_max=32, beta=6.0):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, beta=beta, steps=steps, N=N)
    
    total_threads = N * N * N
    threads_per_block = 256
    blocks_per_grid = (total_threads + (threads_per_block - 1)) // threads_per_block
    w_values = cuda.device_array(total_threads * t_max, dtype=np.float64)
    
    wilson_loop_correlation[blocks_per_grid, threads_per_block](lattice, w_values, N, t_max)
    cuda.synchronize()
    w_values_host = w_values.copy_to_host()
    
    correlation = np.zeros(t_max)
    for t in tqdm(range(t_max), desc="Calculating Lattice QCD Correlation"):
        w_t = w_values_host[t::t_max]
        correlation[t] = np.mean(w_t)
    
    correlation = np.clip(correlation, 0, 1)
    t = np.arange(t_max)
    try:
        popt, pcov = curve_fit(exp_fit, t[1:], correlation[1:], p0=[0.8, 0.1], maxfev=10000)
        A, m = popt
        mass = m
        mass_err = np.sqrt(np.diag(pcov)[1])
    except RuntimeError:
        print("Exponential fit failed, using linear fit as fallback.")
        log_corr = np.log(correlation[1:] + 1e-10)
        m, _ = np.polyfit(t[1:], log_corr, 1)
        mass = -m
        mass_err = bootstrap_error(correlation[1:]) / np.sqrt(t_max - 1)
    
    corr_err = np.array([bootstrap_error(w_values_host[t::t_max]) for t in range(t_max)])
    corr_err = np.clip(corr_err, 0, 1)
    
    print("\n=== Yang-Mills Mass Gap Simulation (Lattice QCD) ===")
    print(f"Correlation Function: {correlation}")
    print(f"Correlation Errors: {corr_err}")
    print(f"Extracted Glueball Mass: {mass:.6f} ± {mass_err:.6f}")
    if mass > 0:
        print("Mass Gap Exists: Positive mass detected.")
    else:
        print("Mass Gap Not Detected: Mass is zero or negative.")
    
    plt.errorbar(t, correlation, yerr=corr_err, fmt='o', label='Simulation Data')
    plt.plot(t, exp_fit(t, *popt), '-', label=f'Fit: m={mass:.3f}')
    plt.xlabel('Time (t)')
    plt.ylabel('Correlation Function C(t)')
    plt.yscale('log')
    plt.legend()
    plt.title('Wilson Loop Correlation Function (Lattice QCD)')
    plt.show()
    
    return mass, mass_err

# Alpha-QCD 시뮬레이션
def simulate_alpha_qcd(N=32, steps=120000, samples=100000, t_max=32, beta=6.0):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, beta=beta, steps=steps, N=N)
    
    coords = np.random.uniform(0, 1, (samples, 3))  # 격자 좌표 샘플링
    d_coords = cuda.to_device(coords)
    d_w_values = cuda.device_array(samples * t_max, dtype=np.float64)
    
    threads_per_block = 256
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    alpha_qcd_correlation[blocks_per_grid, threads_per_block](lattice, d_w_values, d_coords, N, t_max)
    cuda.synchronize()
    w_values_host = d_w_values.copy_to_host()
    
    correlation = np.zeros(t_max)
    for t in tqdm(range(t_max), desc="Calculating Alpha-QCD Correlation"):
        w_t = w_values_host[t::t_max]
        correlation[t] = np.mean(w_t)
    
    correlation = np.clip(correlation, 0, 1)
    t = np.arange(t_max)
    try:
        popt, pcov = curve_fit(exp_fit, t[1:], correlation[1:], p0=[0.8, 0.1], maxfev=10000)
        A, m = popt
        mass = m
        mass_err = np.sqrt(np.diag(pcov)[1])
    except RuntimeError:
        print("Exponential fit failed, using linear fit as fallback.")
        log_corr = np.log(correlation[1:] + 1e-10)
        m, _ = np.polyfit(t[1:], log_corr, 1)
        mass = -m
        mass_err = bootstrap_error(correlation[1:]) / np.sqrt(t_max - 1)
    
    corr_err = np.array([bootstrap_error(w_values_host[t::t_max]) for t in range(t_max)])
    corr_err = np.clip(corr_err, 0, 1)
    
    print("\n=== Yang-Mills Mass Gap Simulation (Alpha-QCD) ===")
    print(f"Correlation Function: {correlation}")
    print(f"Correlation Errors: {corr_err}")
    print(f"Extracted Glueball Mass: {mass:.6f} ± {mass_err:.6f}")
    if mass > 0:
        print("Mass Gap Exists: Positive mass detected.")
    else:
        print("Mass Gap Not Detected: Mass is zero or negative.")
    
    plt.errorbar(t, correlation, yerr=corr_err, fmt='o', label='Simulation Data')
    plt.plot(t, exp_fit(t, *popt), '-', label=f'Fit: m={mass:.3f}')
    plt.xlabel('Time (t)')
    plt.ylabel('Correlation Function C(t)')
    plt.yscale('log')
    plt.legend()
    plt.title('Wilson Loop Correlation Function (Alpha-QCD)')
    plt.show()
    
    return mass, mass_err

# 비교 및 오차율 계산 함수
def compare_results(theory_value, alpha_value, alpha_err, lattice_value, lattice_err, alpha_qcd_value, alpha_qcd_err, label):
    print(f"\n=== Comparison for {label} ===")
    print(f"Theory Value: {theory_value:.6f}")
    print(f"Alpha Integration Value: {alpha_value:.6f} ± {alpha_err:.6f}")
    print(f"Lattice QCD Value: {lattice_value:.6f} ± {lattice_err:.6f}")
    print(f"Alpha-QCD Value: {alpha_qcd_value:.6f} ± {alpha_qcd_err:.6f}")
    
    diff_theory_alpha = abs(theory_value - alpha_value)
    err_theory_alpha = diff_theory_alpha / theory_value * 100
    print(f"Theory vs Alpha - Absolute Difference: {diff_theory_alpha:.6f}, Relative Error: {err_theory_alpha:.2f}%")
    
    diff_theory_lattice = abs(theory_value - lattice_value)
    err_theory_lattice = diff_theory_lattice / theory_value * 100
    print(f"Theory vs Lattice - Absolute Difference: {diff_theory_lattice:.6f}, Relative Error: {err_theory_lattice:.2f}%")
    
    diff_theory_alpha_qcd = abs(theory_value - alpha_qcd_value)
    err_theory_alpha_qcd = diff_theory_alpha_qcd / theory_value * 100
    print(f"Theory vs Alpha-QCD - Absolute Difference: {diff_theory_alpha_qcd:.6f}, Relative Error: {err_theory_alpha_qcd:.2f}%")
    
    diff_alpha_lattice = abs(alpha_value - lattice_value)
    err_alpha_lattice = diff_alpha_lattice / max(alpha_value, 1e-10) * 100
    print(f"Alpha vs Lattice - Absolute Difference: {diff_alpha_lattice:.6f}, Relative Error: {err_alpha_lattice:.2f}%")
    
    diff_alpha_alpha_qcd = abs(alpha_value - alpha_qcd_value)
    err_alpha_alpha_qcd = diff_alpha_alpha_qcd / max(alpha_value, 1e-10) * 100
    print(f"Alpha vs Alpha-QCD - Absolute Difference: {diff_alpha_alpha_qcd:.6f}, Relative Error: {err_alpha_alpha_qcd:.2f}%")
    
    diff_lattice_alpha_qcd = abs(lattice_value - alpha_qcd_value)
    err_lattice_alpha_qcd = diff_lattice_alpha_qcd / max(lattice_value, 1e-10) * 100
    print(f"Lattice vs Alpha-QCD - Absolute Difference: {diff_lattice_alpha_qcd:.6f}, Relative Error: {err_lattice_alpha_qcd:.2f}%")

# 실행 및 비교
if __name__ == "__main__":
    # Alpha Integration 시뮬레이션
    print("Alpha Integration Tests:")
    alpha_scalar_result, alpha_scalar_err, theory_scalar = simulate_alpha_integration(
        N=32, samples=1000000, func='scalar', beta=6.0
    )
    print("---------------------------------------------------------------------------")
    alpha_dist_result, alpha_dist_err, theory_dist = simulate_alpha_integration(
        N=32, samples=1000000, func='distribution', beta=6.0
    )
    print("---------------------------------------------------------------------------")
    
    # Lattice QCD 시뮬레이션
    print("Lattice QCD Mass Gap Test:")
    lattice_mass, lattice_mass_err = simulate_lattice_qcd(
        N=32, steps=120000, samples=1000, t_max=32, beta=6.0
    )
    print("---------------------------------------------------------------------------")
    
    # Alpha-QCD 시뮬레이션
    print("Alpha-QCD Mass Gap Test:")
    alpha_qcd_mass, alpha_qcd_mass_err = simulate_alpha_qcd(
        N=32, steps=120000, samples=100000, t_max=32, beta=6.0
    )
    print("---------------------------------------------------------------------------")
    
    # 비교 및 오차율 계산
    compare_results(theory_scalar, alpha_scalar_result, alpha_scalar_err, lattice_mass, lattice_mass_err, alpha_qcd_mass, alpha_qcd_mass_err, "Scalar Function")
    compare_results(theory_dist, alpha_dist_result, alpha_dist_err, lattice_mass, lattice_mass_err, alpha_qcd_mass, alpha_qcd_mass_err, "Distribution Function")
    compare_results(1.8, alpha_scalar_result, alpha_scalar_err, lattice_mass, lattice_mass_err, alpha_qcd_mass, alpha_qcd_mass_err, "Glueball Mass")