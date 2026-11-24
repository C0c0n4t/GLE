import numpy as np
from typing import Callable, Tuple, List, Dict
import numpy as np
from numpy import typing as npt
from scipy.fft import fft, ifft, fftfreq

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams["text.usetex"] = True
# import plotly.graph_objects as go

from tqdm import tqdm

from helper import Grid

# def lawson4_step(u, v, dz, L_fourier, N_func, w):
#     """
#     u, v: текущие поля в физическом пространстве (массивы)
#     dz: шаг по z
#     L_fourier: функция от w, возвращающая значение линейного оператора L(w)
#     N_func: функция, возвращающая (Nu, Nv) = N(u, v) в физическом пространстве
#     w: массив частот (w = 2 * np.pi * fftfreq(N, d=dt))
#     """

#     # 1. Вычисляем фазовые множители (экспоненты) для линейной части
#     # L(w) - диагональный оператор в Фурье-пространстве
#     Lw = L_fourier(w)
#     E_half = np.exp(0.5 * dz * Lw)  # exp(dz/2 * L)
#     E_full = np.exp(dz * Lw)        # exp(dz * L)

#     # 2. Переводим начальные поля в Фурье-пространство
#     u_hat = fft(u)
#     v_hat = fft(v)

#     # --- Стадия 1 (k1) ---
#     Nu1, Nv1 = N_func(u, v)
#     Nu1_hat = fft(Nu1)
#     Nv1_hat = fft(Nv1)

#     # --- Стадия 2 (k2) ---
#     # u_n2 = E_half * (u_n + dz/2 * N(u_n))
#     # В Фурье-пространстве сложение и умножение на скаляр поэлементные
#     u_n2_hat = E_half * (u_hat + 0.5 * dz * Nu1_hat)
#     v_n2_hat = E_half * (v_hat + 0.5 * dz * Nv1_hat)

#     # Переводим для вычисления нелинейности
#     u_n2 = ifft(u_n2_hat)
#     v_n2 = ifft(v_n2_hat)
#     Nu2, Nv2 = N_func(u_n2, v_n2)
#     Nu2_hat = fft(Nu2)
#     Nv2_hat = fft(Nv2)

#     # --- Стадия 3 (k3) ---
#     # u_n3 = E_half * u_n + dz/2 * N(u_n2)
#     u_n3_hat = E_half * u_hat + 0.5 * dz * Nu2_hat
#     v_n3_hat = E_half * v_hat + 0.5 * dz * Nv2_hat

#     u_n3 = ifft(u_n3_hat)
#     v_n3 = ifft(v_n3_hat)
#     Nu3, Nv3 = N_func(u_n3, v_n3)
#     Nu3_hat = fft(Nu3)
#     Nv3_hat = fft(Nv3)

#     # --- Стадия 4 (k4) ---
#     # u_n4 = E_full * u_n + dz * E_half * N(u_n3)
#     u_n4_hat = E_full * u_hat + dz * E_half * Nu3_hat
#     v_n4_hat = E_full * v_hat + dz * E_half * Nv3_hat

#     u_n4 = ifft(u_n4_hat)
#     v_n4 = ifft(v_n4_hat)
#     Nu4, Nv4 = N_func(u_n4, v_n4)
#     Nu4_hat = fft(Nu4)
#     Nv4_hat = fft(Nv4)

#     # --- Финальный шаг ---
#     # u_{n+1} = E_full * (u_n + dz/6 * N(u_n))
#     #         + dz/3 * E_half * (N(u_n2) + N(u_n3))
#     #         + dz/6 * N(u_n4)
#     u_next_hat = (
#         E_full * (u_hat + dz/6 * Nu1_hat)
#         + (dz/3) * E_half * (Nu2_hat + Nu3_hat)
#         + (dz/6) * Nu4_hat
#     )
#     v_next_hat = (
#         E_full * (v_hat + dz/6 * Nv1_hat)
#         + (dz/3) * E_half * (Nv2_hat + Nv3_hat)
#         + (dz/6) * Nv4_hat
#     )

#     # Возвращаем в физическое пространство
#     return np.real(ifft(u_next_hat)), np.real(ifft(v_next_hat))


class PairedEquations:
    def __init__(
        self,
        delta_beta: float,
        beta2: float,
        delta: float,
        gamma: float,
        Omega_g: float,
        g: float,
    ) -> None:
        self.delta_beta = delta_beta
        self.beta2 = beta2
        self.delta = delta
        self.gamma = gamma
        self.Omega_g = Omega_g
        self.g = g

        # self.L = lambda u, v: 1j * delta_beta * u / 2 + delta
        self.L_fourier = lambda w: g * (1 - w**2 / Omega_g**2) / 2 + 1j * (
            delta_beta / 2 - delta * w + beta2 * w**2 / 2
        )
        N_temp = (
            lambda u, v: 1j
            * gamma
            * (
                (np.abs(u) ** 2 + 2 / 3 * np.abs(v) ** 2) * u
                + 1 / 3 * v**2 * np.conj(u)
            )
        )
        self.N = lambda u, v: (N_temp(u, v), N_temp(v, u))

    @classmethod
    def from_params(cls, params: Dict) -> "PairedEquations":
        """Create a model from a dictionary of paired-equation parameters."""
        required = ("delta_beta", "beta2", "delta", "gamma", "Omega_g", "g")
        missing = [name for name in required if name not in params]
        if missing:
            raise KeyError(f"Missing paired-equation parameters: {', '.join(missing)}")

        return cls(
            delta_beta=params["delta_beta"],
            beta2=params["beta2"],
            delta=params["delta"],
            gamma=params["gamma"],
            Omega_g=params["Omega_g"],
            g=params["g"],
        )

    def to_params(self) -> Dict:
        return {
            "delta_beta": self.delta_beta,
            "beta2": self.beta2,
            "delta": self.delta,
            "gamma": self.gamma,
            "Omega_g": self.Omega_g,
            "g": self.g,
        }

    def _nonlin_step(self, u, v, dz):
        # overflow for dz > 0.01
        # k1_u, k1_v = self.N(u, v)
        # k2_u, k2_v = self.N(u + 0.5 * dz * k1_u, v + 0.5 * dz * k1_v)
        # return u + dz * k2_u, v + dz * k2_v,

        k1_u, k1_v = self.N(u, v)
        k2_u, k2_v = self.N(u + 0.5 * dz * k1_u, v + 0.5 * dz * k1_v)
        k3_u, k3_v = self.N(u + 0.5 * dz * k2_u, v + 0.5 * dz * k2_v)
        k4_u, k4_v = self.N(u + dz * k3_u, v + dz * k3_v)

        return u + (dz / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u), v + (dz / 6.0) * (
            k1_v + 2 * k2_v + 2 * k3_v + k4_v
        )

    def _lin_step(self, u, v, dz):
        u_hat, v_hat = fft(u), fft(v)
        u_hat, v_hat = self.expL_fourier(u_hat, dz), self.expL_fourier(v_hat, dz)
        return ifft(u_hat), ifft(v_hat)

    def _lie(self, u, v, dz):
        u, v = self._lin_step(u, v, dz)
        return self._nonlin_step(u, v, dz)

    def _strang(self, u, v, dz):
        u, v = self._lin_step(u, v, dz / 2)
        u, v = self._nonlin_step(u, v, dz)
        return self._lin_step(u, v, dz / 2)

    def _N(self, span, d) -> int:
        return int(np.floor((span[1] - span[0]) / d) + 1)

    def solve(
        self,
        t_grid: Grid,
        z_grid: Grid,
        initial_cond: Tuple[np.ndarray, np.ndarray],
        method="",
    ):
        u, v = initial_cond
        yield u, v, z_grid.points[0]

        # choose method
        w = 2 * np.pi * fftfreq(len(t_grid), d=t_grid.h)
        # FIXME bad, but fast
        self.expL_fourier = lambda u, dz: np.exp(self.L_fourier(w) * dz) * u

        for z in tqdm(z_grid.points[1:]):
            u, v = self._strang(
                u,
                v,
                z_grid.h,
            )
            yield u, v, z

    def get_rolled_t_grid(self, t_grid, z, c0, w0):
        """
        Переставляет элементы t_grid так, чтобы солитон оказался в центре.
        Сохраняет абсолютные координаты точек.
        """
        L_t = t_grid[-1] - t_grid[0]
        dt = t_grid[1] - t_grid[0]

        # Положение центра солитона
        t_center = (z - w0) / c0

        # На сколько шагов сетки нужно сдвинуть
        N = len(t_grid)
        center_idx = N // 2
        soliton_idx = int(np.round((t_center - t_grid[0]) / dt))
        shift = center_idx - soliton_idx

        return np.roll(t_grid, shift)

    def bright_soliton_analytical(
        self,
        t_grid: np.ndarray,
        z: npt.NDArray[np.float64] | float,
        params: dict,
        moving_frame=True,
        middle=True,
        mod_error: List|None=None
    ):
        # just params for analytical solution
        k = params["k"]
        c0 = params["c0"]
        omega = params["omega"]
        w0 = params.get("w0", 0.0)

        L_t = t_grid[-1] - t_grid[0]
        t0 = t_grid[0]

        xi1 = (self.beta2 * omega**2 - 2 * omega / c0 + 2 * k - self.delta_beta) / (
            self.beta2 * c0**2
        )
        xi2 = -2 * self.gamma / (self.beta2 * c0**2)

        if xi1 >= 0:
            raise ValueError(f"$xi1$ is wrong: {xi1}")
        if xi2 <= 0:
            raise ValueError(f"xi2 is wrong: {xi2}")

        eta = np.sqrt(-xi1)
        A = np.sqrt(-2.0 * xi1 / xi2)

        t = t_grid
        if isinstance(z, np.ndarray):
            # z at axis 1
            z, t = np.meshgrid(z, t)

        # move t frame to z
        if moving_frame:

            # to have proper segment we should set low = t0
            if middle:
                low = (z - w0) / c0 + (-L_t / 2)
            else:
                low = (z - w0) / c0 + t0

            temp = t + low
            # try with roll
            t = low + np.mod(t - low, L_t)


            if mod_error is not None:

                print(temp[:, 20])
                print(t[:, 20])

                if len(t.shape) == 1:
                    t = t.reshape(-1, 1)
                    temp = temp.reshape(-1, 1)

                max_t_grid_shift_by_z = np.max(
                    np.abs(np.sort(t.T % L_t, axis=1) - np.sort(temp.T % L_t, axis=1))
                )
                mod_error.append(max_t_grid_shift_by_z)

        w = z - c0 * t - w0
        arg = eta * w
        envelope = A / np.cosh(arg)

        phase = np.exp(1j * (k * z - omega * t))
        u_exact = envelope * phase
        # print(np.all(u_exact == old))
        v_exact = np.zeros_like(envelope)

        return u_exact, v_exact