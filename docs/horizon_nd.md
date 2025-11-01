# nD Horizons and Surface Gravity

This module estimates analog horizons and surface gravity in 2D/3D grids where
the flow velocity is a vector field and the sound speed is a scalar field.

Key definitions
- Horizon: zero level set of f(x) = |v(x)| − c_s(x).
- Unit normal: n = ∇f / |∇f|.
- Acoustic‑exact κ in nD: κ(x_H) = |n · ∇(c_s^2 − |v|^2)| / (2 c_H), where c_H is
  the sound speed on the surface.

Usage (API)
```python
from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd

grids = [x, y]                # or [x, y, z]
v_field = np.stack([vx, vy], axis=-1)  # (..., D), components last
c_s = cs                      # (...,)
surf = find_horizon_surface_nd(grids, v_field, c_s, scan_axis=0)
```

Return values
- `surf.positions`: (N, D) array of horizon points
- `surf.kappa`: (N,) κ values at those points
- `surf.normals`: (N, D) surface normals
- `surf.c_h`: sound speed at the surface

Notes
- This implementation performs a line-scan along a chosen axis (default `x`) and
  linearly interpolates zero crossings. It is intended for synthetic tanh‑like
  profiles and unit tests—not as a production mesher.
- Gradients are computed via central differences with grid spacing inferred from
  the coordinate arrays.

Demo
```bash
# 2D sheet
python scripts/run_horizon_nd_demo.py --dim 2 --nx 160 --ny 40 --sigma 4e-7 \
  --v0 2.0e6 --cs0 1.0e6 --x0 5e-6

# 3D sheet
python scripts/run_horizon_nd_demo.py --dim 3 --nx 64 --ny 24 --nz 16 --sigma 6e-7 \
  --v0 1.8e6 --cs0 1.0e6 --x0 5e-6
```

Outputs are saved under `results/horizon_nd_demo/` with JSON summary and, for 2D,
a PNG plot of the horizon overlay.

Future extensions
- Robust level‑set or marching cubes/squares surface extraction for complex flows
- Patch‑wise graybody via local normal profiles
- Optional CuPy acceleration for large grids
- Direct OpenPMD→nD pipeline with vector fields from PIC output

