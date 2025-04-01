import cupy as cp
import numpy as np

# SU(N) 행렬 생성 (CuPy에서 사용)
def su_n_matrix(angles, N):
    U = cp.eye(N, dtype=cp.complex128)
    for i in range(min(N, len(angles))):
        U[i, i] = cp.cos(angles[i]) + 1j * cp.sin(angles[i])
    return U

# 격자 초기화 (CuPy에서 생성)
def initialize_lattice(N_size, D, N_su):
    lattice_shape = tuple([N_size] * D) + (D, N_su, N_su)
    lattice = cp.zeros(lattice_shape, dtype=cp.complex128)
    for idx in cp.ndindex(lattice_shape[:-3]):
        for mu in range(D):
            angles = cp.random.uniform(0, 0.1, N_su)
            lattice[idx + (mu,)] = su_n_matrix(angles, N_su)
    return lattice

# Metropolis-Hastings 업데이트 (CuPy 커널)
metropolis_kernel = cp.RawKernel(r'''
extern "C" __global__
void metropolis_update(
    const double* lattice, double* transformed, double delta, double beta, 
    int N_size, int D, int N_su) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N_size;
    for (int d = 1; d < D; d++) total_elements *= N_size;
    total_elements *= D;
    
    if (idx < total_elements) {
        int mu = idx % D;
        int flat_idx = idx / D;
        int coords[4];
        int temp_idx = flat_idx;
        for (int d = D - 1; d >= 0; d--) {
            coords[d] = temp_idx % N_size;
            temp_idx /= N_size;
        }
        
        // Staple 계산
        double complex staple[10][10];
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                staple[i][j] = 0.0 + 0.0 * I;
            }
        }
        
        for (int nu = 0; nu < D; nu++) {
            if (nu != mu) {
                int coords_plus[4];
                for (int d = 0; d < D; d++) coords_plus[d] = coords[d];
                coords_plus[nu] = (coords[nu] + 1) % N_size;
                
                int idx_base = 0;
                int idx_plus_base = 0;
                for (int d = D - 1; d >= 0; d--) {
                    idx_base = idx_base * N_size + coords[d];
                    idx_plus_base = idx_plus_base * N_size + coords_plus[d];
                }
                
                for (int i = 0; i < N_su; i++) {
                    for (int j = 0; j < N_su; j++) {
                        for (int k = 0; k < N_su; k++) {
                            int idx_nu = idx_base * D * N_su * N_su + nu * N_su * N_su + i * N_su + k;
                            int idx_plus_nu = idx_plus_base * D * N_su * N_su + nu * N_su * N_su + j * N_su + k;
                            staple[i][j] += lattice[idx_nu] * conj(lattice[idx_plus_nu]);
                        }
                    }
                }
            }
        }
        
        // U_old
        double complex U_old[10][10];
        int idx_mu = 0;
        for (int d = D - 1; d >= 0; d--) idx_mu = idx_mu * N_size + coords[d];
        idx_mu = idx_mu * D * N_su * N_su + mu * N_su * N_su;
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                U_old[i][j] = lattice[idx_mu + i * N_su + j];
            }
        }
        
        // U_new 생성
        double complex U_new[10][10];
        double r = ((double)threadIdx.x / blockDim.x); // 간단한 난수 대체
        for (int i = 0; i < N_su; i++) {
            double angle = r * 2 * delta - delta;
            for (int j = 0; j < N_su; j++) {
                if (i == j) {
                    U_new[i][j] = cos(angle) + I * sin(angle);
                } else {
                    U_new[i][j] = 0.0 + 0.0 * I;
                }
            }
            r = fmod(r * 1103515245 + 12345, 2147483647) / 2147483647.0; // LCG 난수
        }
        
        double complex U_temp[10][10];
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                U_temp[i][j] = 0.0 + 0.0 * I;
                for (int k = 0; k < N_su; k++) {
                    U_temp[i][j] += U_new[i][k] * U_old[k][j];
                }
            }
        }
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                U_new[i][j] = U_temp[i][j];
            }
        }
        
        // 액션 변화 계산
        double dS = 0.0;
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                dS += creal(staple[i][j] * (U_old[j][i] - U_new[j][i]));
            }
        }
        dS *= beta;
        
        // Metropolis 수락
        double r_accept = fmod(r * 1103515245 + 12345, 2147483647) / 2147483647.0;
        if (dS < 0 || r_accept < exp(-dS)) {
            for (int i = 0; i < N_su; i++) {
                for (int j = 0; j < N_su; j++) {
                    transformed[idx_mu + i * N_su + j] = U_new[i][j];
                }
            }
        } else {
            for (int i = 0; i < N_su; i++) {
                for (int j = 0; j < N_su; j++) {
                    transformed[idx_mu + i * N_su + j] = U_old[i][j];
                }
            }
        }
    }
}
''', 'metropolis_update')

# 열역학 업데이트 실행
def thermalize_lattice(lattice, beta=6.0, steps=1000, N_size=16, D=4, N_su=3):
    total_threads = N_size ** D * D
    threads_per_block = 256
    blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
    
    lattice_flat = lattice.ravel()
    transformed = cp.copy(lattice_flat)
    for step in range(steps):
        metropolis_kernel((blocks_per_grid,), (threads_per_block,), 
                         (lattice_flat, transformed, 0.1, beta, N_size, D, N_su))
        cp.cuda.Stream.null.synchronize()
        lattice_flat = cp.copy(transformed)
        if step % 100 == 0:
            print(f"Thermalization Step {step}/{steps}")
    return lattice_flat.reshape(lattice.shape)

# 사용자 정의 함수 시뮬레이션
def simulate_custom_function(user_func, N_size=16, D=4, N_su=3, samples=1000000, a=-1, b=1):
    lattice = initialize_lattice(N_size, D, N_su)
    lattice = thermalize_lattice(lattice, N_size=N_size, D=D, N_su=N_su)
    coords = cp.random.uniform(a, b, (samples, D))
    f_values = cp.array([user_func(x) for x in coords.get()], dtype=cp.float64)
    
    integral = cp.mean(f_values) * (b - a)
    integral_err = cp.std(f_values) / cp.sqrt(samples) * (b - a)
    L_gamma = (b - a) * cp.sqrt(D)
    
    result = L_gamma * integral
    print(f"=== Custom Function: {user_func.__name__} ===")
    print(f"Simulation Result: {float(result):.6f} ± {float(L_gamma * integral_err):.6f}")
    return result, L_gamma * integral_err

# CUDA Wilson 루프 시뮬레이션
wilson_kernel = cp.RawKernel(r'''
extern "C" __global__
void wilson_loop(
    const double* lattice, double* w_values, const double* coords, 
    int N_size, int D, int N_su) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < coords[0]) {  // coords[0]은 샘플 수
        int x = (int)coords[idx * 2];
        int t = (int)coords[idx * 2 + 1];
        
        double complex loop[10][10];
        for (int i = 0; i < N_su; i++) {
            for (int j = 0; j < N_su; j++) {
                loop[i][j] = (i == j) ? 1.0 + 0.0 * I : 0.0 + 0.0 * I;
            }
        }
        
        double complex temp[10][10];
        for (int i = 0; i < 4; i++) {
            for (int m = 0; m < N_su; m++) {
                for (int n = 0; n < N_su; n++) {
                    temp[m][n] = 0.0 + 0.0 * I;
                    if (x + i < N_size) {
                        int idx_base = x + i;
                        for (int d = 1; d < D - 1; d++) idx_base *= N_size;
                        idx_base = (idx_base * N_size + t) * D * N_su * N_su;
                        for (int k = 0; k < N_su; k++) {
                            temp[m][n] += loop[m][k] * lattice[idx_base + k * N_su + n];
                        }
                    }
                }
            }
            for (int m = 0; m < N_su; m++) {
                for (int n = 0; n < N_su; n++) {
                    loop[m][n] = temp[m][n];
                }
            }
        }
        
        double trace = 0.0;
        for (int i = 0; i < N_su; i++) {
            trace += creal(loop[i][i]);
        }
        w_values[idx] = trace / N_su;
    }
}
''', 'wilson_loop')

def simulate_gauge_invariance(N_size=16, D=4, N_su=3, samples=10000):
    lattice = initialize_lattice(N_size, D, N_su)
    lattice = thermalize_lattice(lattice, N_size=N_size, D=D, N_su=N_su)
    
    # 게이지 변환
    transformed = cp.copy(lattice)
    coords = cp.array([[x, y, z, t] for x in range(N_size) for y in range(N_size) 
                       for z in range(N_size) for t in range(N_size)])
    theta_func = lambda x: cp.array([0.1 * cp.sum(x[:D])] * N_su)
    theta_values = cp.array([theta_func(coord) for coord in coords])
    U = cp.array([su_n_matrix(theta, N_su) for theta in theta_values])
    U_dag = cp.conj(U.transpose(0, 2, 1))
    for mu in range(D):
        transformed[..., mu, :, :] = cp.einsum('ijk,ikl,ilm->ijm', U, lattice[..., mu, :, :], U_dag)
    
    # Wilson 루프 계산
    coords = cp.random.uniform(0, N_size, (samples, 2))
    coords_with_size = cp.concatenate((cp.array([samples], dtype=cp.float64), coords.ravel()))
    d_coords = coords_with_size
    d_w_values = cp.zeros(samples, dtype=cp.float64)
    
    threads_per_block = 256
    blocks_per_grid = (samples + (threads_per_block - 1)) // threads_per_block
    lattice_flat = lattice.ravel()
    wilson_kernel((blocks_per_grid,), (threads_per_block,), 
                  (lattice_flat, d_w_values, d_coords, N_size, D, N_su))
    cp.cuda.Stream.null.synchronize()
    w_values_before = d_w_values.get()
    result_before = np.mean(w_values_before)
    err_before = np.std(w_values_before) / np.sqrt(samples)
    
    transformed_flat = transformed.ravel()
    wilson_kernel((blocks_per_grid,), (threads_per_block,), 
                  (transformed_flat, d_w_values, d_coords, N_size, D, N_su))
    cp.cuda.Stream.null.synchronize()
    w_values_after = d_w_values.get()
    result_after = np.mean(w_values_after)
    err_after = np.std(w_values_after) / np.sqrt(samples)
    
    print("=== Gauge Invariance Test (Wilson Loop) ===")
    print(f"Before Transformation: {result_before:.6f} ± {err_before:.6f}")
    print(f"After Transformation: {result_after:.6f} ± {err_after:.6f}")
    print(f"Difference: {abs(result_before - result_after):.6f}")
    print(f"Relative Difference: {abs(result_before - result_after) / result_before * 100:.2f}%")

# 사용자 정의 함수 예시
def f_scalar(x):
    return x[0] * x[1]  # 예: x_1 * x_2

def f_distribution(x):
    return 1 / (abs(x[0]) + 1e-10)  # 예: 1/|x|

# 실행 예시
if __name__ == "__main__":
    # 4D, SU(3) 격자에서 f(x_1, x_2) = x_1 x_2 시뮬레이션
    simulate_custom_function(f_scalar, N_size=16, D=4, N_su=3)
    print("---------------------------------------------------------------------------")
    
    # 4D, SU(3) 격자에서 f(x) = 1/x 시뮬레이션
    simulate_custom_function(f_distribution, N_size=16, D=4, N_su=3)
    print("---------------------------------------------------------------------------")
    
    # 4D, SU(3) 격자에서 게이지 불변성 테스트
    simulate_gauge_invariance(N_size=16, D=4, N_su=3)