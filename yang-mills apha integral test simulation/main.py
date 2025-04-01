import numpy as np
import cupy as cp
from scipy import integrate, optimize
from scipy.linalg import expm
import matplotlib.pyplot as plt
import logging
from typing import Callable, Tuple, List

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상수 정의
LAMBDA_QCD = 0.213  # GeV
G_COUPLING = 1.0    # 결합 상수 g
SIGMA_LATTICE = np.array([0.04, 0.05])  # 격자 QCD 범위 (GeV^2)
N_SU = 3            # SU(3)
HBAR_C = 0.19732698 # GeV·fm (ħc)
THEORY_VALUES = {'E_0': 0.213, 'sigma': 0.0454, 'gamma': 0.470}
N_SAMPLES = 1000000 # GPU 샘플 수

# SU(3) Gell-Mann 행렬 및 구조 상수
def su3_generators() -> List[np.ndarray]:
    logger.info("SU(3) 생성자 초기화")
    return [
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)
    ]

def su3_structure_constants() -> np.ndarray:
    logger.info("SU(3) 구조 상수 계산")
    generators = su3_generators()
    f = np.zeros((8, 8, 8))
    for a in range(8):
        for b in range(8):
            comm = generators[a] @ generators[b] - generators[b] @ generators[a]
            for c in range(8):
                f[a, b, c] = np.imag(np.trace(comm @ generators[c])) / 2
    return f

# Alpha Integration 클래스
class AlphaIntegration:
    def __init__(self, f: Callable, gamma: Callable, mu: Callable, a: float, b: float, dim: int = 4):
        self.f = f
        self.gamma = gamma
        self.mu = mu
        self.a = a
        self.b = b
        self.dim = dim
        logger.info(f"Alpha Integration 초기화: a={a}, b={b}, dim={dim}")

    def sequential_integration(self, x: np.ndarray, k: int) -> float:
        if k == 0:
            return self.f(x)
        def integrand(t):
            x_new = x.copy()
            x_new[k-1] = t
            return self.sequential_integration(x_new, k-1)
        result, error = integrate.quad(integrand, -np.inf, x[k-1], limit=1000)
        logger.debug(f"순차 적분 k={k}, 결과={result}, 오차={error}")
        return result

    def compute_uai(self, s_values: np.ndarray) -> float:
        integrand = lambda s: self.f(self.gamma(s)) * self.mu(s)
        result, error = integrate.quad(integrand, self.a, self.b, limit=1000)
        logger.info(f"UAI 계산 완료: 결과={result}, 오차={error}")
        return result

# Yang-Mills 시뮬레이터
class YangMillsSimulator:
    def __init__(self, lambda_qcd: float = LAMBDA_QCD, g: float = G_COUPLING):
        self.lambda_qcd = lambda_qcd
        self.g = g
        self.generators = su3_generators()
        self.f_abc = su3_structure_constants()
        self.n_generators = len(self.generators)
        self.generators_cp = cp.array(self.generators, dtype=cp.complex128)
        self.f_abc_cp = cp.array(self.f_abc, dtype=cp.float64)
        logger.info("Yang-Mills 시뮬레이터 초기화")

    def gauge_field(self, x: np.ndarray, a: int, mu: int, params: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        alpha, beta = params
        r = np.sqrt(np.sum(x[:3]**2))
        return alpha * self.lambda_qcd * np.exp(-beta * r) * self.generators[a]

    def field_strength(self, x: np.ndarray, params: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
        h = 1e-5
        F = np.zeros((4, 4, self.n_generators), dtype=complex)
        for mu in range(4):
            for nu in range(4):
                if mu < nu:
                    x_plus_mu = x.copy(); x_plus_mu[mu] += h
                    x_plus_nu = x.copy(); x_plus_nu[nu] += h
                    for a in range(self.n_generators):
                        dmu_A_nu = (self.gauge_field(x_plus_mu, a, nu, params) - 
                                   self.gauge_field(x, a, nu, params)) / h
                        dnu_A_mu = (self.gauge_field(x_plus_nu, a, mu, params) - 
                                   self.gauge_field(x, a, mu, params)) / h
                        A_mu_a = self.gauge_field(x, a, mu, params)
                        A_nu_a = self.gauge_field(x, a, nu, params)
                        comm = self.g * (A_mu_a @ A_nu_a - A_nu_a @ A_mu_a)
                        F[mu, nu, a] = dmu_A_nu - dnu_A_mu + comm
        return F

    def string_tension(self) -> float:
        m = self.lambda_qcd
        k_max = 5 * m
        integrand = lambda k: 4 * np.pi * k**2 / ((2 * np.pi)**3) * 1 / (k**2 + m**2)
        sigma, error = integrate.quad(integrand, 0, k_max, limit=1000)
        logger.info(f"끈 장력 계산: sigma={sigma}, 오차={error}")
        return m**2 * (1 + sigma / m)

    def mass_gap(self) -> float:
        sigma = self.string_tension()
        E_0 = np.sqrt(sigma)
        logger.info(f"질량 간격 계산: E_0={E_0}")
        return E_0

    def compute_D_A_squared_gpu(self, x: cp.ndarray, alpha: float, beta: float) -> cp.ndarray:
        """CuPy를 사용한 GPU 계산"""
        N_samples = x.shape[0]
        D_A_sq = cp.zeros(N_samples, dtype=cp.float64)
        h = 1e-5
        
        logger.info(f"CuPy GPU 계산 시작: 샘플 수={N_samples}")
        r = cp.sqrt(cp.sum(x[:, :3]**2, axis=1))
        for idx in range(N_samples):
            x_vec = x[idx]
            D_A_sq_temp = 0.0
            for i in range(3):
                for j in range(3):
                    for a in range(self.n_generators):
                        x_plus = x_vec.copy()
                        x_plus[i] += h
                        r_plus = cp.sqrt(cp.sum(x_plus[:3]**2))
                        A = alpha * self.lambda_qcd * cp.exp(-beta * r[idx]) * self.generators_cp[a]
                        A_plus = alpha * self.lambda_qcd * cp.exp(-beta * r_plus) * self.generators_cp[a]
                        dA = (A_plus - A) / h
                        commutator = cp.zeros((3, 3), dtype=cp.complex128)
                        for b in range(self.n_generators):
                            for c in range(self.n_generators):
                                f_val = self.f_abc_cp[a, b, c]
                                A_i_b = alpha * self.lambda_qcd * cp.exp(-beta * r[idx]) * self.generators_cp[b]
                                A_j_c = alpha * self.lambda_qcd * cp.exp(-beta * r[idx]) * self.generators_cp[c]
                                commutator += f_val * (A_i_b @ A_j_c)
                        D_A = dA + self.g * commutator
                        D_A_sq_temp += cp.sum(cp.abs(D_A)**2)
            D_A_sq[idx] = D_A_sq_temp
        return D_A_sq

    def gribov_parameter(self) -> float:
        def D_A_squared(params: np.ndarray):
            alpha, beta = params
            L = 1.0  # GeV^-1
            N_samples = N_SAMPLES
            x = cp.random.uniform(-L/2, L/2, (N_samples, 4)).astype(cp.float64)
            
            logger.info(f"CuPy GPU 계산 활성화: alpha={alpha}, beta={beta}")
            D_A_sq = self.compute_D_A_squared_gpu(x, alpha, beta)
            result = cp.mean(D_A_sq) / (L**4)
            logger.debug(f"D_A_squared 중간 결과: {float(result.get())}")
            return float(result.get())

        logger.info("Gribov 파라미터 최적화 시작")
        result = optimize.minimize(D_A_squared, x0=[1.0, 1.0], bounds=[(0, 10), (0, 10)], method='L-BFGS-B')
        gamma_sq = result.fun
        gamma = np.sqrt(gamma_sq)
        logger.info(f"Gribov 파라미터 계산 완료: gamma={gamma}, 최적화 성공={result.success}")
        return gamma

    def wilson_loop(self, L: float, T: float) -> float:
        sigma = self.string_tension()
        s_values = np.linspace(0, L, 100)
        path = lambda s: np.array([s, s, s, s * T / L])
        A_integral = 0
        for s in s_values:
            x = path(s)
            for mu in range(4):
                for a in range(self.n_generators):
                    A_integral += self.g * self.gauge_field(x, a, mu) * (L / 100)
        W = np.trace(expm(1j * A_integral))
        logger.debug(f"Wilson 루프 중간 계산: W={W}")
        return np.abs(W) * np.exp(-sigma * L * T)

    def gauge_invariance_test(self, L: float, T: float) -> Tuple[float, float]:
        W_before = self.wilson_loop(L, T)
        U = expm(1j * self.generators[0])
        def transformed_field(x, a, mu, params=(1.0, 1.0)):
            A = self.gauge_field(x, a, mu, params)
            return U @ A @ U.T.conj()
        s_values = np.linspace(0, L, 100)
        path = lambda s: np.array([s, s, s, s * T / L])
        A_integral = 0
        for s in s_values:
            x = path(s)
            for mu in range(4):
                for a in range(self.n_generators):
                    A_integral += self.g * transformed_field(x, a, mu) * (L / 100)
        W_after = np.abs(np.trace(expm(1j * A_integral))) * np.exp(-self.string_tension() * L * T)
        return W_before, W_after

# 검증 및 비교 함수
def compare_results(sim_results: dict, theory_values: dict):
    print("\n=== 결과 비교 ===")
    print(f"{'항목':<15} {'이론 예측':<15} {'시뮬레이션':<15} {'격자 범위':<20}")
    print("-" * 65)
    for key in theory_values:
        theory = theory_values[key]
        sim = sim_results.get(key, 0)
        diff_sim = abs(theory - sim) / theory * 100 if theory != 0 else 0
        lattice_range = f"{SIGMA_LATTICE[0]:.4f}-{SIGMA_LATTICE[1]:.4f}" if key == 'sigma' else "-"
        print(f"{key:<15} {theory:<15.4f} {sim:<15.4f} {lattice_range:<20} 시뮬 차이: {diff_sim:.2f}%")

def test_alpha_integration():
    s_values = np.linspace(-1, 1, 1000)
    f1 = lambda x: x[0] * x[1]
    gamma1 = lambda s: np.array([s, s, 0, 0])
    mu1 = lambda s: 1.0
    ai1 = AlphaIntegration(f1, gamma1, mu1, -1, 1)
    result1 = ai1.compute_uai(s_values)
    print(f"Alpha Integration Test (L^1): UAI = {result1:.4f}, Expected ≈ 0.6667")

def test_yang_mills():
    ym = YangMillsSimulator()

    # 시뮬레이션 결과
    sim_results = {
        'E_0': ym.mass_gap(),
        'sigma': ym.string_tension(),
        'gamma': ym.gribov_parameter()
    }

    # 결과 비교
    compare_results(sim_results, THEORY_VALUES)

    # Wilson 루프 및 게이지 불변성 테스트
    L_values = np.linspace(0.1, 2, 100)
    W_values = [ym.wilson_loop(L, 1.0) for L in L_values]
    W_before, W_after = ym.gauge_invariance_test(1.0, 1.0)
    print(f"\n게이지 불변성 테스트: W_before = {W_before:.4f}, W_after = {W_after:.4f}, 차이 = {abs(W_before - W_after):.4f}")

    # 시각화
    plt.plot(L_values, W_values, label='Wilson Loop')
    plt.xlabel('L (fm)')
    plt.ylabel('<W(C)>')
    plt.title('Wilson Loop Decay (Confinement Test)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("=== Alpha Integration Tests ===")
    test_alpha_integration()
    print("\n=== Yang-Mills Tests with CuPy ===")
    test_yang_mills()