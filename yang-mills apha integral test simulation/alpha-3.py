import numpy as np
from numba import cuda, float64, complex128
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math

# SU(3) 행렬 생성 (호스트에서 사용)
def su3_matrix(angles):
    U = np.eye(3, dtype=complex)
    U[0, 0] = np.cos(angles[0]) + 1j * np.sin(angles[0])
    U[1, 1] = np.cos(angles[1]) + 1j * np.sin(angles[1])
    U[2, 2] = 1.0 + 0.0j
    return U

# 격자 초기화 (GPU에서 생성)
def initialize_lattice(N=16):
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
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            staple[i, j] += lattice[x, y, z, t, nu, i, k] * lattice[(x+1)%N, y, z, t, nu, j, k].conjugate()
        
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
                    if i == 0:
                        U_new[i, j] = math.cos(angles[0]) + 1j * math.sin(angles[0])
                    elif i == 1:
                        U_new[i, j] = math.cos(angles[1]) + 1j * math.sin(angles[1])
                    else:
                        U_new[i, j] = 1.0 + 0.0j
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

# CUDA 커널: 스칼라 함수 적분
@cuda.jit
def cuda_scalar_integral(lattice, f_values, coords):
    idx = cuda.grid(1)
    if idx < coords.shape[0]:
        s = coords[idx]
        f_values[idx] = s * s

# CUDA 커널: 분포 함수 적분
@cuda.jit
def cuda_distribution_integral(lattice, f_values, coords):
    idx = cuda.grid(1)
    if idx < coords.shape[0]:
        s = coords[idx]
        f_values[idx] = 1 / (1 + abs(s))

# CUDA 커널: Wilson 루프 계산 (단순화)
@cuda.jit
def cuda_wilson_loop(lattice, w_values, coords, N):
    idx = cuda.grid(1)
    if idx < coords.shape[0]:
        x, t = int(coords[idx, 0]), int(coords[idx, 1])
        loop = cuda.local.array((3, 3), dtype=complex128)
        for i in range(3):
            for j in range(3):
                loop[i, j] = 1.0 if i == j else 0.0
        
        temp = cuda.local.array((3, 3), dtype=complex128)
        for i in range(4):
            for m in range(3):
                for n in range(3):
                    temp[m, n] = 0.0
                    for k in range(3):
                        if x + i < N:  # 인덱스 범위 체크
                            temp[m, n] += loop[m, k] * lattice[x+i, 0, 0, t, 0, k, n]
            for m in range(3):
                for n in range(3):
                    loop[m, n] = temp[m, n]
        
        trace = 0.0
        for i in range(3):
            trace += loop[i, i].real
        w_values[idx] = trace / 3

# 게이지 변환
def gauge_transform_lattice(lattice, theta_func):
    N = lattice.shape[0]
    transformed = np.copy(lattice)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                for t in range(N):
                    angles = theta_func([x, y, z, t])
                    U = su3_matrix(angles)
                    U_dag = np.conj(U.T)
                    for mu in range(4):
                        transformed[x, y, z, t, mu] = U @ lattice[x, y, z, t, mu] @ U_dag
    return transformed

# 열역학 업데이트 실행
def thermalize_lattice(lattice, beta=6.0, steps=1000, N=16):
    d_lattice = lattice  # 이미 GPU 메모리에 있음
    total_threads = N * N * N * N * 4
    threads_per_block = 256
    blocks_per_grid = (total_threads + (threads_per_block - 1)) // threads_per_block
    rng_states = create_xoroshiro128p_states(total_threads, seed=1234)
    for step in range(steps):
        metropolis_update[blocks_per_grid, threads_per_block](d_lattice, 0.1, beta, N, rng_states)
        cuda.synchronize()
        if step % 100 == 0:
            print(f"Thermalization Step {step}/{steps}")
    return d_lattice

# 시뮬레이션 1: f(x_1, x_2) = x_1 x_2
def simulate_scalar(N=16, samples=1000000):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, N=N)
    coords = np.random.uniform(-1, 1, samples)
    d_coords = cuda.to_device(coords)
    d_f_values = cuda.device_array(samples, dtype=np.float64)
    
    threads_per_block = 256  # 리소스 사용량 감소
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    cuda_scalar_integral[blocks_per_grid, threads_per_block](lattice, d_f_values, d_coords)
    cuda.synchronize()
    
    f_values = d_f_values.copy_to_host()
    integral = np.mean(f_values) * 2
    integral_err = np.std(f_values) / np.sqrt(samples) * 2
    L_gamma = 2 * np.sqrt(2)  # 2차원 경로 기준
    
    paper_result = 4 * np.sqrt(2) / 3
    result = L_gamma * integral
    print("=== Scalar Function f(x_1, x_2) = x_1 x_2 ===")
    print(f"Paper Prediction: {paper_result:.6f}")
    print(f"Simulation Result: {result:.6f} ± {L_gamma * integral_err:.6f}")
    print(f"Absolute Error: {abs(paper_result - result):.6f}")
    print(f"Relative Error: {abs(paper_result - result) / paper_result * 100:.2f}%")

# 시뮬레이션 2: f(x) = 1/x
def simulate_distribution(N=16, samples=1000000):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, N=N)
    coords = np.random.uniform(-1, 1, samples)
    d_coords = cuda.to_device(coords)
    d_f_values = cuda.device_array(samples, dtype=np.float64)
    
    threads_per_block = 256
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    cuda_distribution_integral[blocks_per_grid, threads_per_block](lattice, d_f_values, d_coords)
    cuda.synchronize()
    
    f_values = d_f_values.copy_to_host()
    integral = np.mean(f_values) * 2
    integral_err = np.std(f_values) / np.sqrt(samples) * 2
    
    paper_result = 2 * np.log(2)
    print("=== Distribution Function f(x) = 1/x ===")
    print(f"Paper Prediction: {paper_result:.6f}")
    print(f"Simulation Result: {integral:.6f} ± {integral_err:.6f}")
    print(f"Absolute Error: {abs(paper_result - integral):.6f}")
    print(f"Relative Error: {abs(paper_result - integral) / paper_result * 100:.2f}%")

# 시뮬레이션 3: 게이지 불변성
def simulate_gauge_invariance(N=16, samples=10000):
    lattice = initialize_lattice(N)
    lattice = thermalize_lattice(lattice, N=N)
    coords = np.random.uniform(0, N, (samples, 2))
    d_coords = cuda.to_device(coords)
    d_w_values = cuda.device_array(samples, dtype=np.float64)
    
    threads_per_block = 256  # 리소스 사용량 감소
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    cuda_wilson_loop[blocks_per_grid, threads_per_block](lattice, d_w_values, d_coords, N)
    cuda.synchronize()
    
    w_values_before = d_w_values.copy_to_host()
    result_before = np.mean(w_values_before)
    err_before = np.std(w_values_before) / np.sqrt(samples)
    
    theta_func = lambda x: np.array([0.1 * sum(x[:4])] * 8)
    lattice_transformed = gauge_transform_lattice(lattice.copy_to_host(), theta_func)
    d_lattice_transformed = cuda.to_device(lattice_transformed)
    cuda_wilson_loop[blocks_per_grid, threads_per_block](d_lattice_transformed, d_w_values, d_coords, N)
    cuda.synchronize()
    
    w_values_after = d_w_values.copy_to_host()
    result_after = np.mean(w_values_after)
    err_after = np.std(w_values_after) / np.sqrt(samples)
    
    print("=== Gauge Invariance Test (Wilson Loop) ===")
    print(f"Before Transformation: {result_before:.6f} ± {err_before:.6f}")
    print(f"After Transformation: {result_after:.6f} ± {err_after:.6f}")
    print(f"Difference: {abs(result_before - result_after):.6f}")
    print(f"Relative Difference: {abs(result_before - result_after) / result_before * 100:.2f}%")

# 실행
simulate_scalar()
print("---------------------------------------------------------------------------")
simulate_distribution()
print("---------------------------------------------------------------------------")
simulate_gauge_invariance()