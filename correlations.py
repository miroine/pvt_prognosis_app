"""
PVT correlations for black oil, dry gas, and water.
All correlations use field units: psia, °F, °R, scf/STB, rb/STB, cp.
References: McCain "Properties of Petroleum Fluids", Ahmed "Reservoir Engineering Handbook".
"""

import numpy as np


# ============================================================
# OIL CORRELATIONS
# ============================================================
class OilCorrelations:
    def __init__(self, api, gas_sg, T, rs_corr="Standing",
                 bo_corr="Standing", mu_corr="Beggs-Robinson"):
        self.api = api
        self.gamma_o = 141.5 / (131.5 + api)   # oil specific gravity
        self.gamma_g = gas_sg
        self.T = T                              # °F
        self.T_R = T + 460.0                    # °R
        self.rs_corr = rs_corr
        self.bo_corr = bo_corr
        self.mu_corr = mu_corr

    # ---- Bubble point (inverse of Rs correlation at Rs = Rsi) ----
    def bubble_point(self, Rsi):
        if self.rs_corr == "Standing":
            # Pb = 18.2 * [(Rs/gamma_g)^0.83 * 10^(0.00091T - 0.0125 API) - 1.4]
            term = (Rsi / self.gamma_g) ** 0.83 * 10 ** (0.00091 * self.T - 0.0125 * self.api)
            return 18.2 * (term - 1.4)

        elif self.rs_corr == "Vasquez-Beggs":
            if self.api <= 30:
                C1, C2, C3 = 0.0362, 1.0937, 25.7240
            else:
                C1, C2, C3 = 0.0178, 1.1870, 23.9310
            # Rs = C1 * gamma_g * P^C2 * exp(C3 * API/(T+460))
            Pb = (Rsi / (C1 * self.gamma_g * np.exp(C3 * self.api / self.T_R))) ** (1.0 / C2)
            return Pb

        elif self.rs_corr == "Glaso":
            # Glaso: log Pb* = 1.7669 + 1.7447 log F - 0.30218 (log F)^2
            # F = (Rs/gamma_g)^0.816 * T^0.172 / API^0.989
            # solve for F given Rsi, then iterate. Easier: invert via Rs(P).
            return _invert_rs_for_pb(self, Rsi)

        elif self.rs_corr == "Lasater":
            return _invert_rs_for_pb(self, Rsi)

        else:
            raise ValueError(self.rs_corr)

    # ---- Solution GOR Rs(P) for P <= Pb ----
    def solution_gor(self, P):
        if self.rs_corr == "Standing":
            x = 0.0125 * self.api - 0.00091 * self.T
            return self.gamma_g * ((P / 18.2 + 1.4) * 10 ** x) ** 1.2048

        elif self.rs_corr == "Vasquez-Beggs":
            if self.api <= 30:
                C1, C2, C3 = 0.0362, 1.0937, 25.7240
            else:
                C1, C2, C3 = 0.0178, 1.1870, 23.9310
            return C1 * self.gamma_g * P ** C2 * np.exp(C3 * self.api / self.T_R)

        elif self.rs_corr == "Glaso":
            # log Pb* = 1.7669 + 1.7447 log F - 0.30218 (log F)^2 ; solve for F
            logP = np.log10(P)
            # quadratic: -0.30218 x^2 + 1.7447 x + (1.7669 - logP) = 0  where x = log F
            a, b, c = -0.30218, 1.7447, 1.7669 - logP
            disc = b * b - 4 * a * c
            if disc < 0:
                disc = 0
            logF = (-b + np.sqrt(disc)) / (2 * a)
            F = 10 ** logF
            # F = (Rs/gamma_g)^0.816 * T^0.172 / API^0.989
            Rs = self.gamma_g * (F * self.api ** 0.989 / self.T ** 0.172) ** (1 / 0.816)
            return Rs

        elif self.rs_corr == "Lasater":
            # Simplified Lasater via gas mole fraction
            # yg = (P * gamma_g) / (P*gamma_g + 7.92*T_R)  approximate
            yg = (P * self.gamma_g / self.T_R) / (P * self.gamma_g / self.T_R + 7.0)
            yg = min(max(yg, 0), 0.95)
            # Effective oil MW
            if self.api <= 40:
                Mo = 630 - 10 * self.api
            else:
                Mo = 73110 * self.api ** -1.562
            Rs = (132755 * self.gamma_o * yg) / (Mo * (1 - yg))
            return max(Rs, 0)

    # ---- Bo ----
    def formation_volume_factor(self, P, Rs, saturated=True, Pb=None):
        if self.bo_corr == "Standing":
            F = Rs * (self.gamma_g / self.gamma_o) ** 0.5 + 1.25 * self.T
            Bob = 0.9759 + 1.2e-4 * F ** 1.2

        elif self.bo_corr == "Vasquez-Beggs":
            if self.api <= 30:
                C1, C2, C3 = 4.677e-4, 1.751e-5, -1.811e-8
            else:
                C1, C2, C3 = 4.670e-4, 1.100e-5, 1.337e-9
            Bob = 1.0 + C1 * Rs + (self.T - 60) * (self.api / self.gamma_g) * (C2 + C3 * Rs)

        elif self.bo_corr == "Glaso":
            Bob_star = Rs * (self.gamma_g / self.gamma_o) ** 0.526 + 0.968 * self.T
            logBob = -6.58511 + 2.91329 * np.log10(Bob_star) - 0.27683 * (np.log10(Bob_star)) ** 2
            Bob = 1.0 + 10 ** logBob
        else:
            raise ValueError(self.bo_corr)

        if saturated:
            return Bob

        # Under-saturated: Bo = Bob * exp(-Co * (P - Pb))
        Co = self.oil_compressibility(P, Rs)
        return Bob * np.exp(-Co * (P - Pb))

    def oil_compressibility(self, P, Rs):
        # Vasquez-Beggs Co
        Co = (-1433 + 5 * Rs + 17.2 * self.T - 1180 * self.gamma_g + 12.61 * self.api) / (1e5 * P)
        return max(Co, 1e-6)

    # ---- Viscosity ----
    def dead_oil_viscosity(self):
        if self.mu_corr == "Beggs-Robinson":
            Z = 3.0324 - 0.02023 * self.api
            Y = 10 ** Z
            X = Y * self.T ** -1.163
            return 10 ** X - 1
        elif self.mu_corr == "Beal":
            a = 10 ** (0.43 + 8.33 / self.api)
            return (0.32 + 1.8e7 / self.api ** 4.53) * (360 / (self.T + 200)) ** a
        elif self.mu_corr == "Glaso":
            return 3.141e10 * self.T ** -3.444 * (np.log10(self.api)) ** (10.313 * np.log10(self.T) - 36.447)
        else:
            raise ValueError(self.mu_corr)

    def viscosity(self, P, Rs, Pb, saturated=True):
        mu_od = self.dead_oil_viscosity()
        # Beggs-Robinson live oil
        a = 10.715 * (Rs + 100) ** -0.515
        b = 5.44 * (Rs + 150) ** -0.338
        mu_ob = a * mu_od ** b

        if saturated:
            return mu_ob

        # Vasquez-Beggs under-saturated
        m = 2.6 * P ** 1.187 * np.exp(-11.513 - 8.98e-5 * P)
        return mu_ob * (P / Pb) ** m


def _invert_rs_for_pb(oil, Rsi, P_low=14.7, P_high=15000):
    """Bisection: find P where Rs(P) = Rsi."""
    f_low = oil.solution_gor(P_low) - Rsi
    f_high = oil.solution_gor(P_high) - Rsi
    if f_low * f_high > 0:
        return P_high if abs(f_high) < abs(f_low) else P_low
    for _ in range(60):
        Pm = 0.5 * (P_low + P_high)
        fm = oil.solution_gor(Pm) - Rsi
        if abs(fm) < 1e-3:
            return Pm
        if f_low * fm < 0:
            P_high = Pm
        else:
            P_low = Pm
            f_low = fm
    return 0.5 * (P_low + P_high)


# ============================================================
# GAS CORRELATIONS
# ============================================================
class GasCorrelations:
    def __init__(self, gas_sg, T, N2=0, CO2=0, H2S=0,
                 z_corr="Hall-Yarborough", mu_corr="Lee-Gonzalez-Eakin"):
        self.gamma_g = gas_sg
        self.T = T
        self.T_R = T + 460.0
        self.N2 = N2; self.CO2 = CO2; self.H2S = H2S
        self.z_corr = z_corr
        self.mu_corr = mu_corr

        # Sutton pseudo-criticals
        self.Tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg ** 2
        self.Ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg ** 2

        # Wichert-Aziz correction for sour gas
        if (CO2 + H2S) > 0:
            A = CO2 + H2S
            B = H2S
            eps = 120 * (A ** 0.9 - A ** 1.6) + 15 * (B ** 0.5 - B ** 4)
            Tpc_corr = self.Tpc - eps
            Ppc_corr = self.Ppc * Tpc_corr / (self.Tpc + B * (1 - B) * eps)
            self.Tpc = Tpc_corr
            self.Ppc = Ppc_corr

    def z_factor(self, P):
        Tpr = self.T_R / self.Tpc
        Ppr = P / self.Ppc

        if self.z_corr == "Hall-Yarborough":
            return _z_hall_yarborough(Tpr, Ppr)
        elif self.z_corr == "Dranchuk-Abou-Kassem":
            return _z_dak(Tpr, Ppr)
        else:
            raise ValueError(self.z_corr)

    def formation_volume_factor(self, P, Z):
        # Bg in rb/scf = 0.00504 * Z*T_R/P
        return 0.00504 * Z * self.T_R / P

    def density(self, P, Z):
        # rho_g (lb/ft3) = P*M / (Z*R*T_R), R = 10.732
        M = 28.97 * self.gamma_g
        return P * M / (Z * 10.732 * self.T_R)

    def viscosity(self, P, Z):
        if self.mu_corr == "Lee-Gonzalez-Eakin":
            M = 28.97 * self.gamma_g
            rho = self.density(P, Z) / 62.428  # to g/cc
            K = ((9.4 + 0.02 * M) * self.T_R ** 1.5) / (209 + 19 * M + self.T_R)
            X = 3.5 + 986 / self.T_R + 0.01 * M
            Y = 2.4 - 0.2 * X
            return 1e-4 * K * np.exp(X * rho ** Y)

        elif self.mu_corr == "Carr-Kobayashi-Burrows":
            # Simplified: atmospheric viscosity then ratio correction
            mu1 = (1.709e-5 - 2.062e-6 * self.gamma_g) * self.T + 8.188e-3 \
                  - 6.15e-3 * np.log10(self.gamma_g)
            # Standing pressure correction (approximation)
            Tpr = self.T_R / self.Tpc
            Ppr = P / self.Ppc
            ratio = np.exp(0.01 * Ppr) * (1 + 0.05 * Ppr / Tpr)
            return mu1 * ratio
        else:
            raise ValueError(self.mu_corr)


def _z_hall_yarborough(Tpr, Ppr, tol=1e-8, maxit=100):
    t = 1.0 / Tpr
    A = 0.06125 * t * np.exp(-1.2 * (1 - t) ** 2)
    B = t * (14.76 - 9.76 * t + 4.58 * t ** 2)
    C = t * (90.7 - 242.2 * t + 42.4 * t ** 2)
    D = 2.18 + 2.82 * t

    y = 0.001
    for _ in range(maxit):
        f = (-A * Ppr + (y + y ** 2 + y ** 3 - y ** 4) / (1 - y) ** 3
             - B * y ** 2 + C * y ** D)
        df = ((1 + 4 * y + 4 * y ** 2 - 4 * y ** 3 + y ** 4) / (1 - y) ** 4
              - 2 * B * y + C * D * y ** (D - 1))
        dy = f / df
        y -= dy
        if abs(dy) < tol:
            break
    Z = A * Ppr / y
    return Z


def _z_dak(Tpr, Ppr, tol=1e-8, maxit=100):
    A = [0.3265, -1.0700, -0.5339, 0.01569, -0.05165,
         0.5475, -0.7361, 0.1844, 0.1056, 0.6134, 0.7210]
    rho = 0.27 * Ppr / Tpr  # initial
    Z = 1.0
    for _ in range(maxit):
        c1 = A[0] + A[1] / Tpr + A[2] / Tpr ** 3 + A[3] / Tpr ** 4 + A[4] / Tpr ** 5
        c2 = A[5] + A[6] / Tpr + A[7] / Tpr ** 2
        c3 = A[8] * (A[6] / Tpr + A[7] / Tpr ** 2)
        c4 = A[9] * (1 + A[10] * rho ** 2) * (rho ** 2 / Tpr ** 3) * np.exp(-A[10] * rho ** 2)
        Z_new = 1 + c1 * rho + c2 * rho ** 2 - c3 * rho ** 5 + c4
        rho_new = 0.27 * Ppr / (Z_new * Tpr)
        if abs(rho_new - rho) < tol:
            return Z_new
        rho = rho_new
        Z = Z_new
    return Z


# ============================================================
# WATER CORRELATIONS  (McCain, Meehan, Numbere, Spivey)
# ============================================================
class WaterCorrelations:
    """
    Water PVT properties.
    All correlations accept salinity in ppm (NaCl-equivalent) and T in °F.
    Outputs: Bw [rb/STB], Cw [1/psi], mu_w [cP], rho_w [lb/ft3].
    Includes optional dissolved-gas effect on Bw via Rsw (McCain).
    """

    def __init__(self, salinity_ppm, T, corr="McCain", include_gas=False, Pb=None):
        self.salinity_ppm = salinity_ppm
        self.salinity_wt_pct = salinity_ppm / 1e4   # weight percent
        self.S = salinity_ppm / 1e6                 # mass fraction
        self.T = T
        self.T_R = T + 460.0
        self.corr = corr
        self.include_gas = include_gas
        self.Pb = Pb if Pb is not None else 14.7

    # ---- Solution gas-water ratio Rsw [scf/STB] (McCain) ----
    def rsw(self, P):
        # Pure-water Rsw, then salinity correction
        A = 8.15839 - 6.12265e-2 * self.T + 1.91663e-4 * self.T ** 2 - 2.1654e-7 * self.T ** 3
        B = 1.01021e-2 - 7.44241e-5 * self.T + 3.05553e-7 * self.T ** 2 - 2.94883e-10 * self.T ** 3
        C = (-9.02505 + 0.130237 * self.T - 8.53425e-4 * self.T ** 2
             + 2.34122e-6 * self.T ** 3 - 2.37049e-9 * self.T ** 4) * 1e-7
        Rsw_pure = A + B * P + C * P ** 2
        # Salinity correction (McCain)
        log_ratio = -0.0840655 * self.salinity_wt_pct * self.T ** -0.285854
        return Rsw_pure * 10 ** log_ratio

    # ---- Bw ----
    def bw(self, P):
        if self.corr == "McCain":
            dVwT = -1.0001e-2 + 1.33391e-4 * self.T + 5.50654e-7 * self.T ** 2
            dVwP = (-1.95301e-9 * P * self.T - 1.72834e-13 * P ** 2 * self.T
                    - 3.58922e-7 * P - 2.25341e-10 * P ** 2)
            Bw = (1 + dVwT) * (1 + dVwP)

        elif self.corr == "Meehan":
            # Meehan (1980) - simple, gas-free
            A = 0.9947 + 5.8e-6 * self.T + 1.02e-6 * self.T ** 2
            B = -4.228e-6 + 1.8376e-8 * self.T - 6.77e-11 * self.T ** 2
            C = 1.3e-10 - 1.3855e-12 * self.T + 4.285e-15 * self.T ** 2
            Bw_pure = A + B * P + C * P ** 2
            # Salinity correction
            Bw = Bw_pure * (1 + self.salinity_wt_pct * 1e-4 *
                            (5.1e-8 * P + (5.47e-6 - 1.95e-10 * P) * (self.T - 60)
                             + (-3.23e-8 + 8.5e-13 * P) * (self.T - 60) ** 2))

        elif self.corr == "Numbere":
            # Numbere et al. - close to McCain, slightly different coefficients
            dVwT = -1.0001e-2 + 1.33391e-4 * self.T + 5.50654e-7 * self.T ** 2
            dVwP = -1.95301e-9 * P * self.T - 1.72834e-13 * P ** 2 * self.T \
                   - 3.58922e-7 * P - 2.25341e-10 * P ** 2
            Bw_pure = (1 + dVwT) * (1 + dVwP)
            # Numbere salinity volume correction
            Bw = Bw_pure * (1.0 + self.S * (0.51 - 5.79e-4 * (self.T - 60)))

        elif self.corr == "Spivey":
            # Spivey, Valko, McCain (2004) - more accurate over wider P,T range
            # Tc-form expressions for pure water density, then salinity & pressure correction
            Tc = (self.T - 32) / 1.8       # to °C
            Pmpa = P * 0.00689476           # psia -> MPa
            # Pure water density at (T,P) from polynomial fit (g/cc)
            rho_w_p_pure = (-0.127213 * (Tc / 100) ** 2 + 0.645486 * (Tc / 100) + 1.03265) / \
                           (-0.070291 * (Tc / 100) ** 2 + 0.639589 * (Tc / 100) + 1)
            Ew = (4.221 * (Tc / 100) ** 2 - 3.478 * (Tc / 100) + 6.221) / \
                 (0.5182 * (Tc / 100) ** 2 - 0.4405 * (Tc / 100) + 1)
            Fw = (-11.403 * (Tc / 100) ** 2 + 29.932 * (Tc / 100) + 27.952) / \
                 (0.20684 * (Tc / 100) ** 2 + 0.3768 * (Tc / 100) + 1)
            Iw = (1 / Ew) * np.log(abs(Ew * (Pmpa / 70.0) + Fw))
            Iw_ref = (1 / Ew) * np.log(abs(Ew * (0.1 / 70.0) + Fw))
            rho_w_TP = rho_w_p_pure * np.exp(Iw - Iw_ref)
            # Reservoir water density with salinity (very simplified additive)
            rho_brine = rho_w_TP + 0.668 * self.S + 0.44 * self.S ** 2
            # Bw vs surface (60°F, 14.7 psia, with same salinity)
            rho_brine_sc = 1.0 + 0.668 * self.S + 0.44 * self.S ** 2
            Bw = rho_brine_sc / rho_brine
        else:
            raise ValueError(self.corr)

        # Add dissolved-gas swelling for P <= Pb
        if self.include_gas and P <= self.Pb:
            Rsw = self.rsw(P)
            # ~1.2e-4 rb/STB per scf/STB (McCain approx)
            Bw = Bw + 1.2e-4 * Rsw
        return Bw

    def compressibility(self, P):
        # McCain (gas-free) baseline
        Cw_pure = 1.0 / (7.033 * P + 541.5 * self.salinity_wt_pct
                         - 537 * self.T + 403300)
        if self.include_gas:
            # Dodson-Standing gas-saturated correction
            Rsw = self.rsw(P)
            Cw_pure *= (1 + 8.9e-3 * Rsw)
        return max(Cw_pure, 1e-9)

    def viscosity(self, P):
        S = self.salinity_wt_pct
        A = 109.574 - 8.40564 * S + 0.313314 * S ** 2 + 8.72213e-3 * S ** 3
        B = (-1.12166 + 2.63951e-2 * S - 6.79461e-4 * S ** 2
             - 5.47119e-5 * S ** 3 + 1.55586e-6 * S ** 4)
        mu_w_1atm = A * self.T ** B
        # Pressure correction (McCain)
        return mu_w_1atm * (0.9994 + 4.0295e-5 * P + 3.1062e-9 * P ** 2)

    def density(self, P):
        # Surface brine density approx 62.4*(1 + 0.695*S) lb/ft3
        rho_sc = 62.428 * (1.0 + 0.695 * self.S)
        return rho_sc / self.bw(P)


# ============================================================
# WET GAS / GAS CONDENSATE CORRELATIONS
# ============================================================
class WetGasCorrelations:
    """
    Wet gas properties: gas Z, Bg, mu_g plus vaporised oil-gas ratio Rv (STB/scf).
    Computes a 'reservoir-gas' specific gravity that combines surface gas + condensate
    contribution, then uses standard Sutton + HY/DAK for Z.

    Inputs:
        gas_sg     : surface (separator) gas specific gravity
        api_cond   : condensate API
        cgr        : condensate-gas ratio at separator (STB/MMscf)
        T          : reservoir temperature (°F)
        rv_corr    : 'Constant' (Rv = CGR everywhere, lean gas) or
                     'Linear-Pdew' (Rv decreases linearly with P below Pdew)
        Pdew       : dew-point pressure (psia)  [needed for Linear-Pdew]
    """

    def __init__(self, gas_sg, api_cond, cgr_stb_per_mmscf, T,
                 N2=0, CO2=0, H2S=0,
                 z_corr="Hall-Yarborough", mu_corr="Lee-Gonzalez-Eakin",
                 rv_corr="Linear-Pdew", Pdew=None):
        self.gamma_g_sep = gas_sg
        self.api_cond = api_cond
        self.gamma_cond = 141.5 / (131.5 + api_cond)
        self.cgr = cgr_stb_per_mmscf       # STB / MMscf
        self.Rv_max = cgr_stb_per_mmscf / 1e6   # STB / scf
        self.T = T
        self.T_R = T + 460.0
        self.N2 = N2; self.CO2 = CO2; self.H2S = H2S
        self.z_corr = z_corr
        self.mu_corr = mu_corr
        self.rv_corr = rv_corr
        self.Pdew = Pdew if Pdew else 5000.0

        # ----- Reservoir-fluid (wet) gas SG -----
        # Mo (condensate molecular weight, Cragoe approx)
        Mo = 6084 / (api_cond - 5.9)
        # Gas-equivalent of condensate, Veq (scf/STB) ~ 133300*gamma_o/Mo
        Veq = 133300 * self.gamma_cond / Mo
        # GOR at separator (scf/STB) = 1e6 / CGR
        GOR = 1e6 / cgr_stb_per_mmscf if cgr_stb_per_mmscf > 0 else 1e9
        # Recombined reservoir-fluid SG (McCain):
        self.gamma_g_res = (GOR * gas_sg + 4584 * self.gamma_cond) / (GOR + Veq)

        # ----- Pseudo-criticals using reservoir-fluid SG -----
        sg = self.gamma_g_res
        self.Tpc = 169.2 + 349.5 * sg - 74.0 * sg ** 2
        self.Ppc = 756.8 - 131.0 * sg - 3.6 * sg ** 2

        if (CO2 + H2S) > 0:
            A = CO2 + H2S; B = H2S
            eps = 120 * (A ** 0.9 - A ** 1.6) + 15 * (B ** 0.5 - B ** 4)
            Tpc_corr = self.Tpc - eps
            Ppc_corr = self.Ppc * Tpc_corr / (self.Tpc + B * (1 - B) * eps)
            self.Tpc = Tpc_corr; self.Ppc = Ppc_corr

    def z_factor(self, P):
        Tpr = self.T_R / self.Tpc
        Ppr = P / self.Ppc
        if self.z_corr == "Hall-Yarborough":
            return _z_hall_yarborough(Tpr, Ppr)
        return _z_dak(Tpr, Ppr)

    def formation_volume_factor(self, P, Z):
        return 0.00504 * Z * self.T_R / P

    def density(self, P, Z):
        M = 28.97 * self.gamma_g_res
        return P * M / (Z * 10.732 * self.T_R)

    def viscosity(self, P, Z):
        if self.mu_corr == "Lee-Gonzalez-Eakin":
            M = 28.97 * self.gamma_g_res
            rho = self.density(P, Z) / 62.428
            K = ((9.4 + 0.02 * M) * self.T_R ** 1.5) / (209 + 19 * M + self.T_R)
            X = 3.5 + 986 / self.T_R + 0.01 * M
            Y = 2.4 - 0.2 * X
            return 1e-4 * K * np.exp(X * rho ** Y)
        # Carr-Kobayashi-Burrows
        mu1 = (1.709e-5 - 2.062e-6 * self.gamma_g_res) * self.T + 8.188e-3 \
              - 6.15e-3 * np.log10(self.gamma_g_res)
        Tpr = self.T_R / self.Tpc; Ppr = P / self.Ppc
        return mu1 * np.exp(0.01 * Ppr) * (1 + 0.05 * Ppr / Tpr)

    def rv(self, P):
        """Vaporised oil-gas ratio (STB/scf) at pressure P."""
        if self.rv_corr == "Constant":
            return self.Rv_max
        # Linear in P below Pdew, zero above (above Pdew it stays at Rv_max actually
        # because the gas is single-phase carrying all condensate). Below Pdew the
        # liquid drops out, so Rv decreases.
        if P >= self.Pdew:
            return self.Rv_max
        # Linear from Rv_max @ Pdew to ~0.05*Rv_max at low P
        frac = max(P, 14.7) / self.Pdew
        return self.Rv_max * (0.05 + 0.95 * frac)
