# Streamlit demo: Atoms → Mechanics for single-crystal silicon
# Author: ChatGPT
# Description: Orientation-dependent Young's modulus from elastic tensor → beam deflection.
# Data source for elastic constants: Hopcroft et al., "What is the Young's Modulus of Silicon?" JMEMS (2010)
# Room-temperature cubic elastic constants (GPa): C11=165.7, C12=63.9, C44=79.6

import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="Atoms→Mechanics: Silicon", layout="wide")
st.title("Atoms → Mechanics: Silicon (single crystal)")
st.caption("Orientation-dependent modulus from elastic tensor → beam deflection (Euler–Bernoulli)")

# --- Elastic constants (GPa)
C11, C12, C44 = 165.7, 63.9, 79.6

# --- Compliance components for cubic symmetry
# S11 = (C11 + C12) / ((C11 - C12)*(C11 + 2*C12))
# S12 = -C12 / ((C11 - C12)*(C11 + 2*C12))
# S44 = 1 / C44
den = (C11 - C12) * (C11 + 2*C12)
S11 = (C11 + C12) / den
S12 = -C12 / den
S44 = 1.0 / C44

# VRH isotropic moduli from cubic C_ij
K_V = (C11 + 2*C12) / 3.0
G_V = (C11 - C12 + 3*C44) / 5.0
K_R = 1.0 / (3.0*(S11 + 2.0*S12))
G_R = 5.0 / (4.0*(S11 - S12) + 3.0*S44)
K_H = 0.5*(K_V + K_R)
G_H = 0.5*(G_V + G_R)
E_iso = 9*K_H*G_H/(3*K_H + G_H)
nu_iso = (3*K_H - 2*G_H)/(2*(3*K_H + G_H))


# --- Functions

def E_from_direction(l, m, n, S11, S12, S44):
    norm = math.sqrt(l*l + m*m + n*n)
    l, m, n = l/norm, m/norm, n/norm
    Q = l*l*m*m + m*m*n*n + n*n*l*l
    invE = S11 - 2.0*(S11 - S12 - 0.5*S44)*Q
    return 1.0 / invE  # GPa

# --- Layout
col_left, col_right = st.columns([1,1], gap="large")

with col_left:
    st.subheader("1) Choose orientation")
    quick = st.radio("Quick directions", ["<100>", "<110>", "<111>", "Custom"], horizontal=True, index=1)
    if quick == "<100>":
        l, m, n = 1, 0, 0
    elif quick == "<110>":
        l, m, n = 1, 1, 0
    elif quick == "<111>":
        l, m, n = 1, 1, 1
    else:
        theta = st.slider("Polar angle θ (deg)", 0.0, 180.0, 35.0, 0.1, help="Angle from +z")
        phi = st.slider("Azimuth φ (deg)", 0.0, 360.0, 45.0, 0.1, help="Angle from +x in x–y plane")
        th, ph = math.radians(theta), math.radians(phi)
        # Spherical to Cartesian (z is polar axis)
        l = math.sin(th)*math.cos(ph)
        m = math.sin(th)*math.sin(ph)
        n = math.cos(th)

    E_dir = E_from_direction(l, m, n, S11, S12, S44)
    st.metric("Young's modulus along load direction E(\u27C2)", f"{E_dir:.1f} GPa")

    # Plot E along path from <100> → <111>    # 3D crystal visualization (unit cube and load direction)
    st.markdown("**Crystal & load direction**")
    fig3d = go.Figure()

    BLUE = "#1976D2"; RED = "#D32F2F"; X_COLOR="#E53935"; Y_COLOR="#43A047"; Z_COLOR="#1E88E5"

    # Unit cube wireframe (0..1)
    verts = [(i,j,k) for i in (0,1) for j in (0,1) for k in (0,1)]
    for (x0,y0,z0) in verts:
        for dx,dy,dz in [(1,0,0),(0,1,0),(0,0,1)]:
            x1,y1,z1 = x0+dx, y0+dy, z0+dz
            if x1<=1 and y1<=1 and z1<=1:
                fig3d.add_trace(go.Scatter3d(x=[x0,x1], y=[y0,y1], z=[z0,z1],
                                             mode="lines", line=dict(width=2, color="white"),
                                             showlegend=False, hoverinfo="skip"))

    # Crystal axes
    axis_len = 1.0
    arrow_len = 0.5
    fig3d.add_trace(go.Scatter3d(x=[0,arrow_len], y=[0,0], z=[0,0], mode="lines",
                                 name="[100]", line=dict(width=4, color=X_COLOR)))
    fig3d.add_trace(go.Scatter3d(x=[0,0], y=[0,arrow_len], z=[0,0], mode="lines",
                                 name="[010]", line=dict(width=4, color=Y_COLOR)))
    fig3d.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,arrow_len], mode="lines",
                                 name="[001]", line=dict(width=4, color=Z_COLOR)))

    # Load direction vector n
    Lvec = 1.0
    # Load direction vector n (centered so it always stays inside the cube)
    start = np.array([0.5, 0.5, 0.5])
    d = np.array([l, m, n], dtype=float)
    d = d / max(1e-9, np.linalg.norm(d))
    # Compute max scale to hit a cube face from the center in direction d
    s_candidates = []
    for comp, s0 in zip(d, start):
        if comp > 0:
            s_candidates.append((1.0 - s0)/comp)
        elif comp < 0:
            s_candidates.append((0.0 - s0)/comp)
        else:
            s_candidates.append(np.inf)
    s = min([c for c in s_candidates if c > 0]) * 0.98  # margin
    end = start + s * d

    fig3d.add_trace(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]],
                                 mode="lines", name="n (load dir)",
                                 line=dict(width=12, color=RED)))
    fig3d.add_trace(go.Cone(x=[end[0]], y=[end[1]], z=[end[2]], u=[d[0]], v=[d[1]], w=[d[2]],
                            anchor="tip", sizemode="absolute", sizeref=0.16,
                            showscale=False, colorscale=[[0, RED],[1, RED]], name="load"))

        # Diamond-cubic silicon atoms as filled spheres
    def sphere_mesh(cx, cy, cz, r, nu=14, nv=28, color="#888", opacity=0.95):
        import numpy as _np
        thetas = _np.linspace(0, _np.pi, nu)
        phis = _np.linspace(0, 2*_np.pi, nv, endpoint=False)
        xs, ys, zs = [], [], []
        for th in thetas:
            for ph in phis:
                xs.append(cx + r*_np.sin(th)*_np.cos(ph))
                ys.append(cy + r*_np.sin(th)*_np.sin(ph))
                zs.append(cz + r*_np.cos(th))
        xs = _np.array(xs); ys = _np.array(ys); zs = _np.array(zs)
        i, j, k = [], [], []
        def idx(a,b): return a*len(phis)+b
        for a in range(nu-1):
            for b in range(len(phis)):
                b2 = (b+1) % len(phis)
                i.append(idx(a,b));   j.append(idx(a+1,b)); k.append(idx(a+1,b2))
                i.append(idx(a,b));   j.append(idx(a+1,b2)); k.append(idx(a,b2))
        return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k, color=color, opacity=opacity, name="Si atom", showscale=False)

    # Conventional cell positions (fractional) for diamond cubic Si
    si_A = [(0,0,0),(0,0.5,0.5),(0.5,0,0.5),(0.5,0.5,0)]
    si_B = [(0.25,0.25,0.25),(0.25,0.75,0.75),(0.75,0.25,0.75),(0.75,0.75,0.25)]
    r = 0.10  # sphere radius in fractional cell units
    for x0,y0,z0 in si_A:
        fig3d.add_trace(sphere_mesh(x0,y0,z0,r,color=BLUE,opacity=0.95))
    for x0,y0,z0 in si_B:
        fig3d.add_trace(sphere_mesh(x0,y0,z0,r,color=BLUE,opacity=0.95))

        # Nearest-neighbor bonds (diamond cubic) as thick lines
    import numpy as _np
    A = _np.array(si_A, dtype=float)
    B = _np.array(si_B, dtype=float)
    shifts = _np.array([[i,j,k] for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)], dtype=float)
    seg_keys = set()
    thr = 0.45  # nearest-neighbor distance ~0.433 in fractional units
    for a in A:
        for b in B:
            for s in shifts:
                p2 = b + s
                if (p2>=0).all() and (p2<=1).all():
                    d = float(_np.linalg.norm(p2 - a))
                    if d < thr:
                        p1 = a
                        key = tuple(_np.round(_np.concatenate([p1,p2]),3))
                        keyr = tuple(_np.round(_np.concatenate([p2,p1]),3))
                        if key in seg_keys or keyr in seg_keys:
                            continue
                        seg_keys.add(key)
                        fig3d.add_trace(go.Scatter3d(x=[p1[0],p2[0]], y=[p1[1],p2[1]], z=[p1[2],p2[2]],
                                                     mode="lines", line=dict(width=8, color=BLUE), name="bond", showlegend=False))

        # Add translucent cube faces (white) and (111) plane (blue)
    def _add_face(pts, rgba):
        xs, ys, zs = zip(*pts)
        i = [0,0]; j = [1,2]; k = [2,3]
        fig3d.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                                  color=rgba, opacity=1.0, hoverinfo="skip", showlegend=False))
    for _pts,_c in [
        ([(0,0,0),(0,1,0),(0,1,1),(0,0,1)], "rgba(255,255,255,0.08)"),  # x=0
        ([(1,0,0),(1,1,0),(1,1,1),(1,0,1)], "rgba(255,255,255,0.08)"),  # x=1
        ([(0,0,0),(1,0,0),(1,0,1),(0,0,1)], "rgba(255,255,255,0.08)"),  # y=0
        ([(0,1,0),(1,1,0),(1,1,1),(0,1,1)], "rgba(255,255,255,0.08)"),  # y=1
        ([(0,0,0),(1,0,0),(1,1,0),(0,1,0)], "rgba(255,255,255,0.08)"),  # z=0
        ([(0,0,1),(1,0,1),(1,1,1),(0,1,1)], "rgba(255,255,255,0.08)")   # z=1
    ]:
        _add_face(_pts, _c)

    tri111 = [(1,0,0),(0,1,0),(0,0,1)]
    xs, ys, zs = zip(*tri111)
    fig3d.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, i=[0], j=[1], k=[2],
                              color="rgba(25,118,210,0.18)", opacity=1.0,
                              name="(111) plane", hoverinfo="skip"))

    # Axis cones (arrows)
    cone_style = dict(sizemode="absolute", sizeref=0.15, showscale=False,
                      colorscale=[[0,"#999999"],[1,"#999999"]])
    fig3d.add_trace(go.Cone(x=[arrow_len], y=[0], z=[0], u=[1], v=[0], w=[0], anchor="tip",
             sizemode="absolute", sizeref=0.12, showscale=False,
             colorscale=[[0, X_COLOR],[1, X_COLOR]], name="[100] →"))
    fig3d.add_trace(go.Cone(x=[0], y=[arrow_len], z=[0], u=[0], v=[1], w=[0], anchor="tip",
             sizemode="absolute", sizeref=0.12, showscale=False,
             colorscale=[[0, Y_COLOR],[1, Y_COLOR]], name="[010] →"))
    fig3d.add_trace(go.Cone(x=[0], y=[0], z=[arrow_len], u=[0], v=[0], w=[1], anchor="tip",
             sizemode="absolute", sizeref=0.12, showscale=False,
             colorscale=[[0, Z_COLOR],[1, Z_COLOR]], name="[001] →"))

        # Plot inside-cube path used for E vs t (tips of direction vectors)
    def _slerp(u,v,t):
        u = np.array(u)/np.linalg.norm(u); v = np.array(v)/np.linalg.norm(v)
        dot = float(np.clip(np.dot(u,v), -1.0, 1.0))
        omega = math.acos(dot)
        if abs(omega) < 1e-12:
            return u
        return (math.sin((1-t)*omega)*u + math.sin(t*omega)*v) / math.sin(omega)
    ts_path = np.linspace(0,1,100)
    u0, v0 = (1,0,0), (1,1,1)
    pts = np.array([_slerp(u0, v0, t) for t in ts_path])
    fig3d.add_trace(go.Scatter3d(x=Lvec*pts[:,0], y=Lvec*pts[:,1], z=Lvec*pts[:,2],
                                 mode="lines", name="E–t path",
                                 line=dict(width=5, color=BLUE, dash="dash")))

    fig3d.update_layout(scene_camera=dict(eye=dict(x=1.5, y=0.8, z=1.2)),scene=dict(
                            xaxis_title="[100]", yaxis_title="[010]", zaxis_title="[001]",
                            xaxis=dict(range=[0, 1], zeroline=False),
                            yaxis=dict(range=[0, 1], zeroline=False),
                            zaxis=dict(range=[0, 1], zeroline=False),
                            aspectmode="cube"
                        ),
                        margin=dict(l=10,r=10,t=10,b=10),
                        height=420)
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("**Orientation sweep**: <100> → <111>")
    def slerp(u,v,t):
        u = np.array(u)/np.linalg.norm(u)
        v = np.array(v)/np.linalg.norm(v)
        dot = np.clip(np.dot(u,v), -1.0, 1.0)
        omega = math.acos(dot)
        if abs(omega) < 1e-12:
            return u
        return (math.sin((1-t)*omega)*u + math.sin(t*omega)*v) / math.sin(omega)

    u = (1,0,0)
    v = (1,1,1)
    ts = np.linspace(0,1,101)
    E_path = [E_from_direction(*slerp(u,v,t), S11=S11, S12=S12, S44=S44) for t in ts]

    figE = go.Figure()
    figE.add_trace(go.Scatter(x=ts, y=E_path, mode="lines", name="E vs t"))
    figE.update_layout(
        xaxis_title="t (0 = <100> → 1 = <111>)",
        yaxis_title="E (GPa)",
        legend_title="Legend",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(figE, use_container_width=True)

with col_right:
    st.subheader("2) Beam response (exact deflection curve)")
    st.caption("Simply supported beam, centered point load. Exact Euler–Bernoulli deflection shape (small-deflection)")

    b = st.number_input("Width b (mm)", value=10.0, min_value=0.1, step=0.1)/1000.0
    h = st.number_input("Height h (mm)", value=1.0, min_value=0.05, step=0.05)/1000.0
    L = st.number_input("Span L (mm)", value=100.0, min_value=1.0, step=1.0)/1000.0
    P = st.number_input("Load P (N)", value=10.0, min_value=0.01, step=0.1)

    I = b*h**3/12.0
    E_pa = E_dir*1e9

    # Midspan deflection (should match the exact curve at x=L/2)
    delta_mid = P*L**3/(48.0*E_pa*I)  # meters
    st.metric("Midspan deflection δ", f"{delta_mid*1e3:.3f} mm")

    # Exact deflection curve for a simply supported beam with a point load at midspan
    # Piecewise closed-form (Roark/Table):
    # For 0<=x<=a:   w = P*b*x*(L^2 - b^2 - x^2) / (6 E I L)
    # For a<=x<=L:   w = P*a*(L-x)*(L^2 - a^2 - (L-x)^2) / (6 E I L)
    # with a=L/2, b=L/2 for a centered load.
    def deflection_curve_point_midspan(L, P, E, I, npts=400):
        import numpy as np, math
        x = np.linspace(0.0, L, npts)
        a = L/2.0
        b = L - a
        w = np.empty_like(x)
        left = x <= a
        w[left] = P*b*x[left]*(L**2 - b**2 - x[left]**2)/(6.0*E*I*L)
        xr = x[~left]
        w[~left] = P*a*(L - xr)*(L**2 - a**2 - (L - xr)**2)/(6.0*E*I*L)
        return x, w

    x, w = deflection_curve_point_midspan(L, P, E_pa, I, npts=600)

    # Optional comparison curves
    show_iso = st.checkbox("Show isotropic comparison (VRH E)", value=True)
    show_100 = st.checkbox("Show ⟨100⟩", value=True)
    show_111 = st.checkbox("Show ⟨111⟩", value=True)

    curves = []
    if show_iso:
        E_iso_pa = E_iso*1e9
        curves.append(("VRH isotropic", E_iso_pa))
    if show_100:
        E100 = E_from_direction(1,0,0, S11, S12, S44)*1e9
        curves.append(("⟨100⟩", E100))
    if show_111:
        E111 = E_from_direction(1,1,1, S11, S12, S44)*1e9
        curves.append(("⟨111⟩", E111))

    fig = go.Figure()
    # Current orientation curve
    fig.add_trace(go.Scatter(x=1e3*x, y=1e3*w, mode="lines", name="Current orientation"))
    fig.add_trace(go.Scatter(x=[1e3*L/2.0], y=[1e3*delta_mid], mode="markers", name="δ mid (current)"))

    # Add comparison curves
    for label, Ecomp in curves:
        x_c, w_c = deflection_curve_point_midspan(L, P, Ecomp, I, npts=600)
        fig.add_trace(go.Scatter(x=1e3*x_c, y=1e3*w_c, mode="lines", name=label))
        fig.add_trace(go.Scatter(x=[1e3*L/2.0], y=[1e3*(P*L**3/(48.0*Ecomp*I))], mode="markers", name=f"δ mid ({label})", marker_symbol="x"))

    fig.update_layout(
        xaxis_title="Position x (mm)",
        yaxis_title="Deflection w (mm)",
        legend_title="Legend",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )

    st.plotly_chart(fig, use_container_width=True)

# st.divider()
# st.markdown(
#     """
# **Notes**  
# • Assumes small strains and linear elastic behavior (Euler–Bernoulli).  
# • Orientation formula for cubic crystals: \(1/E = S_{11} - 2\,(S_{11}-S_{12}-S_{44}/2)\,(l^2 m^2 + m^2 n^2 + n^2 l^2)\).  
# • Literature range for Si E: ~130–188 GPa depending on direction (Hopcroft et al., 2010).  
# • VRH values shown for polycrystal comparison.
#     """
# )
