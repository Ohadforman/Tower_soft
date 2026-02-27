# try_numeric_multi_queue_undo_LIVE.py
# streamlit run try_numeric_multi_queue_undo_LIVE.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🧪 Try: Zone Marker (Numeric) — Queue + Undo + LIVE update")

# ----------------------------
# Demo numeric data
# ----------------------------
N = 4000
x = np.linspace(0, 12.0, N)

rng = np.random.default_rng(7)
y1 = np.sin(np.linspace(0, 50, N)) + 0.15 * rng.normal(size=N)
y2 = 0.6 * np.cos(np.linspace(0, 20, N)) + 0.10 * rng.normal(size=N)

df = pd.DataFrame({"x": x, "y1": y1, "y2": y2})

# ----------------------------
# State
# ----------------------------
if "zones" not in st.session_state:
    st.session_state["zones"] = []          # saved zones [(a,b), ...]
if "queued" not in st.session_state:
    st.session_state["queued"] = []         # queued zones [(a,b), ...]
if "preview" not in st.session_state:
    st.session_state["preview"] = None      # current selection (a,b) or None
if "msg" not in st.session_state:
    st.session_state["msg"] = ""
if "_last_sel_sig" not in st.session_state:
    st.session_state["_last_sel_sig"] = None

CHART_KEY = "zone_plot_numeric_queue_live"

# ----------------------------
# Helpers: payload + extract selection.box[0].x
# ----------------------------
def _get_payload(chart_key: str, returned_obj):
    if isinstance(returned_obj, dict):
        return returned_obj
    ss = st.session_state.get(chart_key, None)
    if isinstance(ss, dict):
        return ss
    if hasattr(returned_obj, "selection"):
        try:
            return {"selection": getattr(returned_obj, "selection")}
        except Exception:
            pass
    if hasattr(ss, "selection"):
        try:
            return {"selection": getattr(ss, "selection")}
        except Exception:
            pass
    return None

def extract_x_window(payload):
    # Your Streamlit format: selection.box[0].x = [x0, x1]
    if not isinstance(payload, dict):
        return None
    sel = payload.get("selection", None)
    if not isinstance(sel, dict):
        return None
    box = sel.get("box", None)
    if isinstance(box, list) and box and isinstance(box[0], dict):
        xr = box[0].get("x", None)
        if isinstance(xr, (list, tuple)) and len(xr) >= 2:
            try:
                a = float(xr[0]); b = float(xr[1])
                if b < a:
                    a, b = b, a
                return (a, b)
            except Exception:
                return None
    return None

def _dedup_and_sort(zones):
    out = []
    for a, b in sorted(zones, key=lambda t: (t[0], t[1])):
        if not out:
            out.append((a, b))
            continue
        pa, pb = out[-1]
        if abs(a - pa) < 1e-12 and abs(b - pb) < 1e-12:
            continue
        out.append((a, b))
    return out

def _sig(a, b):
    # stable signature to avoid infinite reruns (round to reduce noise)
    return (round(float(a), 9), round(float(b), 9))

# ----------------------------
# UI options
# ----------------------------
st.subheader("⚙️ Options")
copt1, copt2, copt3 = st.columns([1, 1, 2])
with copt1:
    auto_queue = st.checkbox("Auto-queue every selection", value=False)
with copt2:
    keep_preview = st.checkbox("Keep preview after queue", value=False)
with copt3:
    st.caption("LIVE: after you drag a box, it immediately updates overlays.")

# ----------------------------
# Plot
# ----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["x"], y=df["y1"], mode="lines", name="y1"))
fig.add_trace(go.Scatter(x=df["x"], y=df["y2"], mode="lines", name="y2"))

# Saved zones (green)
for (a, b) in st.session_state["zones"]:
    fig.add_vrect(x0=a, x1=b, fillcolor="green", opacity=0.25, line_width=0)

# Queued zones (orange)
for (a, b) in st.session_state["queued"]:
    fig.add_vrect(x0=a, x1=b, fillcolor="orange", opacity=0.18, line_width=0)

# Preview (blue)
if st.session_state["preview"] is not None:
    a, b = st.session_state["preview"]
    fig.add_vrect(x0=a, x1=b, fillcolor="blue", opacity=0.18, line_width=1, line_dash="dot")

fig.update_layout(
    height=620,
    margin=dict(l=10, r=10, t=40, b=10),
    dragmode="select",
    selectdirection="h",
)
fig.update_xaxes(title_text="X", tickangle=-90)

st.caption("Drag a horizontal window on the plot. It should show LIVE as blue/orange/green.")

returned = st.plotly_chart(
    fig,
    use_container_width=True,
    on_select="rerun",
    selection_mode=("box",),
    key=CHART_KEY,
)

payload = _get_payload(CHART_KEY, returned)
sel_rng = extract_x_window(payload)

# ----------------------------
# LIVE update: if NEW selection arrived, update state and rerun once
# ----------------------------
if sel_rng is not None:
    a, b = sel_rng
    sig = _sig(a, b)

    if sig != st.session_state["_last_sel_sig"]:
        st.session_state["_last_sel_sig"] = sig

        # update preview
        st.session_state["preview"] = (a, b)

        # auto-queue (optional)
        if auto_queue:
            st.session_state["queued"].append((a, b))
            st.session_state["queued"] = _dedup_and_sort(st.session_state["queued"])
            st.session_state["msg"] = f"🟧 Queued ({len(st.session_state['queued'])})"
            if not keep_preview:
                st.session_state["preview"] = None

        # IMPORTANT: immediate rerun so the vrect appears NOW
        st.rerun()

# ----------------------------
# Buttons
# ----------------------------
st.subheader("🟩 Zone Actions")

b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1, 1])

with b1:
    if st.button("➕ Add Preview", use_container_width=True, disabled=(st.session_state["preview"] is None)):
        st.session_state["zones"].append(st.session_state["preview"])
        st.session_state["zones"] = _dedup_and_sort(st.session_state["zones"])
        st.session_state["preview"] = None
        st.session_state["msg"] = f"✅ Added preview (saved: {len(st.session_state['zones'])})"
        st.rerun()

with b2:
    if st.button("🟧 Queue Preview", use_container_width=True, disabled=(st.session_state["preview"] is None)):
        st.session_state["queued"].append(st.session_state["preview"])
        st.session_state["queued"] = _dedup_and_sort(st.session_state["queued"])
        if not keep_preview:
            st.session_state["preview"] = None
        st.session_state["msg"] = f"🟧 Queued (queued: {len(st.session_state['queued'])})"
        st.rerun()

with b3:
    if st.button("✅ Add Queued Zones", use_container_width=True, disabled=(len(st.session_state["queued"]) == 0)):
        st.session_state["zones"].extend(st.session_state["queued"])
        st.session_state["zones"] = _dedup_and_sort(st.session_state["zones"])
        st.session_state["queued"] = []
        st.session_state["msg"] = f"✅ Added queued zones (saved: {len(st.session_state['zones'])})"
        st.rerun()

with b4:
    if st.button("↩️ Undo Last Saved", use_container_width=True, disabled=(len(st.session_state["zones"]) == 0)):
        st.session_state["zones"].pop()
        st.session_state["msg"] = f"↩️ Undone last saved (saved: {len(st.session_state['zones'])})"
        st.rerun()

with b5:
    if st.button("↩️ Undo Last Queued", use_container_width=True, disabled=(len(st.session_state["queued"]) == 0)):
        st.session_state["queued"].pop()
        st.session_state["msg"] = f"↩️ Undone last queued (queued: {len(st.session_state['queued'])})"
        st.rerun()

c6, c7 = st.columns([1, 3])
with c6:
    if st.button("🧹 Clear All", use_container_width=True, disabled=(len(st.session_state["zones"]) == 0 and len(st.session_state["queued"]) == 0 and st.session_state["preview"] is None)):
        st.session_state["zones"] = []
        st.session_state["queued"] = []
        st.session_state["preview"] = None
        st.session_state["_last_sel_sig"] = None
        st.session_state["msg"] = "🧽 Cleared everything"
        st.rerun()
with c7:
    st.info(f"Saved: **{len(st.session_state['zones'])}** | Queued: **{len(st.session_state['queued'])}** | Preview: **{'Yes' if st.session_state['preview'] else 'No'}**")

if st.session_state["msg"]:
    st.success(st.session_state["msg"])

# ----------------------------
# Tables
# ----------------------------
st.subheader("📋 Zones")
cA, cB = st.columns(2)

with cA:
    st.markdown("**🟩 Saved zones**")
    if st.session_state["zones"]:
        st.dataframe(pd.DataFrame(
            [{"Zone": i, "Start": a, "End": b} for i, (a, b) in enumerate(st.session_state["zones"], start=1)]
        ), use_container_width=True, hide_index=True)
    else:
        st.write("None")

with cB:
    st.markdown("**🟧 Queued zones**")
    if st.session_state["queued"]:
        st.dataframe(pd.DataFrame(
            [{"Queue": i, "Start": a, "End": b} for i, (a, b) in enumerate(st.session_state["queued"], start=1)]
        ), use_container_width=True, hide_index=True)
    else:
        st.write("None")

with st.expander("DEBUG payload", expanded=False):
    st.write("returned:", returned)
    st.write("session_state[CHART_KEY]:", st.session_state.get(CHART_KEY, None))
    st.write("payload:", payload)
    st.write("sel_rng:", sel_rng)
    st.write("_last_sel_sig:", st.session_state.get("_last_sel_sig"))