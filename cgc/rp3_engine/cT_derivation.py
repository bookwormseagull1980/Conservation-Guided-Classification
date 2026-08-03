"""c_T strict derivation: one-loop gauge Tmunu TT projection for SU(N) (2026-08-03).

Correct physical setup: c_T is the coefficient of the ONE-LOOP gauge-field
Tmunu two-point function projected onto the transverse-traceless (TT)
sector, normalised by the physical polarisation count.  It appears in

    g2_grav_eff = Z * c_T * (N_c^2) / (16 pi^2) * (k^2 / M_P^2)

as the tensor-structure factor of the single gauge loop coupled to the
emergent graviton.

Standard field-theory derivation (d = 4 Euclidean):
  For a massless vector field the Tmunu is
      T_mu nu = F_mu alpha F_nu alpha - (1/4) delta_mu nu F^2,
  and its ONE-LOOP two-point function in the transverse gauge has the
  TT-projected form
      <T_mu nu T_rho sigma>_TT = (N_c^2 - 1) * c_T
          * (delta_mu rho delta_nu sigma + delta_mu sigma delta_nu rho
             - (2/(d-1)) delta_mu nu delta_rho sigma) * (k^4 / (16 pi^2)).

  The trace-free symmetric tensor structure has norm 2 (for each pair)
  and the (d-2) physical helicities contribute.  Contracting with the
  TT projector and dividing by the normalisation gives the universal
  spin-1 coefficient (see e.g. the spin-1 heat-kernel / TT projection
  in gravitational one-loop computations):

      c_T = (1/2) * (d-2)/(d-1) * d/(d-2) ...  -- simplify below.

  Known values for spin s in d dimensions:
      spin-0: c_T = (d-2)/(4(d-1))        [conformal scalar]
      spin-1: c_T = (d-2)/4               [massless vector]
      spin-2: c_T = (d-2)(d+1)/(4(d-1))   [graviton]
  For d = 4:
      spin-1: c_T = (4-2)/4 = 1/2
  (normalised per physical DOF pair; see Christensen-Duff or the
   standard spin projection table).

  NOTE: the framework's 3/4 = 1/2 * 3/2.  The extra 3/2 = N_c^2/(N_c^2-1)
  for SU(3) (9/8) is NOT 3/2, so the framework value requires the full
  colour factor.  We report the strict spin-1 value and the SU(3)-colour
  corrected value.
"""

D = 4


def c_T_spin1(d: int) -> float:
    """Universal TT-projection coefficient for a massless spin-1 field."""
    return (d - 2) / 4.0


def c_T_spin0(d: int) -> float:
    return (d - 2) / (4.0 * (d - 1))


def c_T_spin2(d: int) -> float:
    return (d - 2) * (d + 1) / (4.0 * (d - 1))


def compute():
    cT1 = c_T_spin1(D)
    print("=" * 60)
    print("  c_T STRICT DERIVATION — one-loop gauge Tmunu TT projection")
    print("=" * 60)
    print(f"  Universal spin-s TT coefficients (d = {D}):")
    print(f"    spin-0 (scalar):   c_T = (d-2)/(4(d-1)) = {c_T_spin0(D):.4f}")
    print(f"    spin-1 (vector):   c_T = (d-2)/4        = {c_T_spin1(D):.4f}")
    print(f"    spin-2 (graviton): c_T = (d-2)(d+1)/(4(d-1)) = {c_T_spin2(D):.4f}")
    print()
    print("  Framework uses c_T = 3/4 = 0.75 for SU(3) gauge fields.")
    print(f"  Strict spin-1 value (per colour field): {cT1:.4f} = 1/2")
    print()
    # colour factor for SU(3): the loop carries N_c^2 - 1 = 8 gluons,
    # and the graviton couples to the singlet combination; the effective
    # coefficient per physical polarisation is c_T * (N_c^2-1)/N_c^2.
    Nc = 3.0
    cT_su3 = cT1 * (Nc**2 - 1) / Nc**2
    print(f"  SU(3) colour-corrected (×8/9): c_T = {cT_su3:.4f}")
    print()
    print("  The framework's 0.75 corresponds to 1/2 × 3/2.")
    print("  The 3/2 = (d+1)/(d-1)? No: 3/2 is not a standard colour/")
    print("  tensor factor — the strict values above are 0.5 (spin-1)")
    print("  and 0.444 (SU(3) colour-corrected).")
    return {
        "spin0": c_T_spin0(D),
        "spin1": c_T_spin1(D),
        "spin2": c_T_spin2(D),
        "su3_corrected": cT_su3,
        "framework": 0.75,
    }


if __name__ == "__main__":
    compute()
