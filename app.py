"""Streamlit front end for the reliability-gated solar flare predictor."""

import io
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import RGFusion, GradCAM, enable_dropout, EUV_CHANNELS, IMG_SIZE

APP_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = APP_DIR / "samples"
MODE_LABEL = {
    "full": "Reliability-gated (ours)",
    "concat": "Concatenation",
    "mag": "Magnetogram only",
    "euv": "EUV only",
}

st.set_page_config(page_title="Solar Flare Predictor", page_icon="☀", layout="wide")


def find_checkpoint():
    for p in (APP_DIR / "rgfusion_checkpoint.pt", APP_DIR.parent / "rgfusion_checkpoint.pt"):
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_models(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    models = {}
    for mode, sd in ckpt["state_dicts"].items():
        m = RGFusion(n_euv=ckpt.get("n_euv", 8)).set_mode(mode)
        m.load_state_dict(sd)
        m.eval()
        models[mode] = m
    return models, ckpt


@st.cache_data(show_spinner=False)
def list_samples():
    if not SAMPLES_DIR.is_dir():
        return []
    out = []
    for d in sorted(SAMPLES_DIR.iterdir()):
        if d.is_dir() and (d / "meta.json").exists():
            out.append(json.loads((d / "meta.json").read_text()))
    return out


def load_gray(path_or_buf, size=IMG_SIZE):
    img = Image.open(path_or_buf).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def sample_tensors(folder):
    folder = Path(folder)
    zero = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
    mag_p = folder / "magnetogram.jpg"
    mag = load_gray(mag_p) if mag_p.exists() else zero
    euv = np.stack([
        load_gray(folder / f"{c}.jpg") if (folder / f"{c}.jpg").exists() else zero
        for c in EUV_CHANNELS
    ])
    return mag[None], euv


def corrupt(euv, level, seed=0):
    level = float(level)
    if level <= 0:
        return euv.astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.random(euv.shape).astype(np.float32)
    return ((1 - level) * euv + level * noise).astype(np.float32)


def predict(model, mag, euv, passes=30):
    mt = torch.from_numpy(np.ascontiguousarray(mag, dtype=np.float32)).unsqueeze(0)
    et = torch.from_numpy(np.ascontiguousarray(euv, dtype=np.float32)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        point = torch.sigmoid(model(mt, et)).item()
    enable_dropout(model)
    with torch.no_grad():
        draws = np.array([torch.sigmoid(model(mt, et)).item() for _ in range(passes)])
    model.eval()
    return point, float(draws.mean()), float(draws.std()), draws


def explain(model, mag, euv):
    mt = torch.from_numpy(np.ascontiguousarray(mag, dtype=np.float32)).unsqueeze(0)
    et = torch.from_numpy(np.ascontiguousarray(euv, dtype=np.float32)).unsqueeze(0)
    cam_m = cam_e = tr_m = tr_e = None
    g1 = GradCAM(model, model.enc_mag.net[-1])
    cam_m = g1(mt, et)
    g1.close()
    g2 = GradCAM(model, model.enc_euv.net[-1])
    cam_e = g2(mt, et)
    g2.close()
    model.eval()
    with torch.no_grad():
        model(mt, et)
    if "mag" in model.last_trust:
        tr_m = model.last_trust["mag"][0, 0].cpu().numpy()
        tr_e = model.last_trust["euv"][0, 0].cpu().numpy()
    return cam_m, cam_e, tr_m, tr_e


def upscale(a, size=IMG_SIZE):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR)) / 255.0


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="none", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


st.title("Solar flare prediction with reliability-gated fusion")
st.caption(
    "Predicts whether a solar active region will produce a C-class flare or larger "
    "within 24 hours, from an HMI magnetogram and eight AIA EUV passbands."
)

ckpt_path = find_checkpoint()
if ckpt_path is None:
    st.error(
        "No `rgfusion_checkpoint.pt` found. Put it in this folder or the one above it."
    )
    st.stop()

models, ckpt = load_models(ckpt_path)
samples = list_samples()

with st.sidebar:
    st.header("Input")
    source = st.radio("Source", ["Sample active region", "Upload your own"], index=0)

    mag = euv = None
    truth = None
    chosen_name = ""

    if source == "Sample active region":
        if not samples:
            st.warning("No samples found in `samples/`.")
        else:
            labels = [f"{s['label_text']}  ({s['noaa']})" for s in samples]
            idx = st.selectbox("Active region", range(len(samples)),
                               format_func=lambda i: labels[i])
            s = samples[idx]
            chosen_name = s["noaa"]
            mag, euv = sample_tensors(SAMPLES_DIR / s["folder"])
            truth = s
    else:
        st.caption("Grayscale images, any size. Missing EUV channels are filled with zeros.")
        up_mag = st.file_uploader("HMI magnetogram", type=["jpg", "jpeg", "png"])
        up_euv = st.file_uploader("AIA EUV passbands (up to 8)", type=["jpg", "jpeg", "png"],
                                  accept_multiple_files=True)
        if up_mag is not None:
            zero = np.zeros((IMG_SIZE, IMG_SIZE), np.float32)
            mag = load_gray(up_mag)[None]
            chans = [load_gray(f) for f in (up_euv or [])][:8]
            while len(chans) < 8:
                chans.append(zero)
            euv = np.stack(chans)
            chosen_name = "uploaded"

    st.header("Model")
    mode = st.selectbox("Variant", list(MODE_LABEL), index=0,
                        format_func=lambda m: MODE_LABEL[m])

    st.header("Instrument failure")
    level = st.slider("EUV corruption", 0.0, 1.0, 0.0, 0.05,
                      help="Interpolates the EUV stack toward uniform noise, "
                           "simulating a data gap, eclipse or saturated detector.")
    passes = st.slider("MC-dropout passes", 5, 60, 30, 5)

if mag is None:
    st.info("Pick a sample active region in the sidebar, or upload a magnetogram.")
    st.stop()

euv_used = corrupt(euv, level)
model = models[mode]
point, mean, std, draws = predict(model, mag, euv_used, passes)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Flare probability", f"{mean:.1%}")
c2.metric("Uncertainty (1 sd)", f"±{std:.1%}")
verdict = "FLARE LIKELY" if mean >= 0.5 else "QUIET"
c3.metric("Verdict at 0.5", verdict)
if truth:
    c4.metric("Ground truth", truth["label_text"],
              help=f"peak flux {truth['peak_flux']:.2e} W/m^2 in the next 24 h")
else:
    c4.metric("Ground truth", "unknown")

if truth:
    correct = (mean >= 0.5) == bool(truth["label"])
    if correct:
        st.success(f"Correct. Observed peak flux was {truth['peak_flux']:.2e} W/m^2.")
    else:
        st.error(f"Wrong. Observed peak flux was {truth['peak_flux']:.2e} W/m^2.")

if level > 0:
    st.warning(
        f"EUV is {level:.0%} corrupted. The report found the gate does **not** protect "
        "against this; compare the gated and concatenation variants to see for yourself."
    )

st.subheader("Inputs")
cols = st.columns(5)
cols[0].image(upscale(mag[0], 256), caption="HMI magnetogram", clamp=True)
for k, name in enumerate(["94", "131", "171", "193"]):
    cols[k + 1].image(upscale(euv_used[EUV_CHANNELS.index(name)], 256),
                      caption=f"AIA {name}", clamp=True)
cols2 = st.columns(5)
for k, name in enumerate(["211", "304", "335", "1700"]):
    cols2[k].image(upscale(euv_used[EUV_CHANNELS.index(name)], 256),
                   caption=f"AIA {name}", clamp=True)

st.subheader("Where the model is looking")
if mode != "full":
    st.caption(
        f"Trust masks exist only in the gated variant. Showing Grad-CAM for "
        f"{MODE_LABEL[mode]}."
    )
cam_m, cam_e, tr_m, tr_e = explain(model, mag, euv_used)

ncol = 3 if tr_m is not None else 2
fig, ax = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.4))
ax = np.atleast_2d(ax)
ax[0, 0].imshow(mag[0], cmap="gray"); ax[0, 0].set_title("Magnetogram", fontsize=10)
ax[0, 1].imshow(mag[0], cmap="gray")
ax[0, 1].imshow(upscale(cam_m), cmap="jet", alpha=0.5)
ax[0, 1].set_title("Grad-CAM (magnetic)", fontsize=10)
ax[1, 0].imshow(euv_used[1], cmap="inferno"); ax[1, 0].set_title("AIA 131", fontsize=10)
ax[1, 1].imshow(euv_used[1], cmap="inferno")
ax[1, 1].imshow(upscale(cam_e), cmap="jet", alpha=0.5)
ax[1, 1].set_title("Grad-CAM (EUV)", fontsize=10)
if tr_m is not None:
    ax[0, 2].imshow(upscale(tr_m), cmap="viridis", vmin=0, vmax=1)
    ax[0, 2].set_title(f"Trust mask (magnetic) mean {tr_m.mean():.2f}", fontsize=10)
    ax[1, 2].imshow(upscale(tr_e), cmap="viridis", vmin=0, vmax=1)
    ax[1, 2].set_title(f"Trust mask (EUV) mean {tr_e.mean():.2f}", fontsize=10)
for a in ax.ravel():
    a.axis("off")
for t in fig.findobj(matplotlib.text.Text):
    t.set_color("#888888")
st.image(fig_to_png(fig))

if tr_m is not None:
    st.caption(
        f"Mean trust: magnetic {tr_m.mean():.2f}, EUV {tr_e.mean():.2f}. "
        "An EUV mask that stays near 1 is not suppressing anything, which is why "
        "corrupting EUV still damages the prediction."
    )

st.subheader("Uncertainty and instrument failure")
left, right = st.columns(2)

with left:
    fig2, a2 = plt.subplots(figsize=(5, 3.1))
    a2.hist(draws, bins=16, color="#4c72b0", edgecolor="none")
    a2.axvline(mean, color="#dd8452", lw=2, label=f"mean {mean:.2f}")
    a2.axvline(0.5, color="#888888", ls="--", lw=1, label="threshold")
    a2.set_xlabel("predicted probability"); a2.set_ylabel("MC passes")
    a2.set_title(f"{passes} MC-dropout draws", fontsize=10)
    a2.legend(fontsize=8)
    for t in fig2.findobj(matplotlib.text.Text):
        t.set_color("#888888")
    a2.tick_params(colors="#888888")
    for sp in a2.spines.values():
        sp.set_color("#888888")
    fig2.patch.set_alpha(0)
    a2.patch.set_alpha(0)
    st.image(fig_to_png(fig2))

with right:
    levels = np.linspace(0, 1, 11)
    curves = {}
    for m_name in ("full", "concat"):
        ys = []
        for lv in levels:
            _, mu, _, _ = predict(models[m_name], mag, corrupt(euv, lv), passes=1)
            ys.append(mu)
        curves[m_name] = ys
    fig3, a3 = plt.subplots(figsize=(5, 3.1))
    a3.plot(levels, curves["full"], "o-", color="#55a868", label="gated")
    a3.plot(levels, curves["concat"], "o--", color="#dd8452", label="concat")
    a3.axvline(level, color="#888888", ls=":", lw=1)
    a3.axhline(0.5, color="#888888", ls="--", lw=1)
    a3.set_xlabel("EUV corruption level"); a3.set_ylabel("flare probability")
    a3.set_title("This region, as EUV degrades", fontsize=10)
    a3.legend(fontsize=8)
    for t in fig3.findobj(matplotlib.text.Text):
        t.set_color("#888888")
    a3.tick_params(colors="#888888")
    for sp in a3.spines.values():
        sp.set_color("#888888")
    fig3.patch.set_alpha(0)
    a3.patch.set_alpha(0)
    st.image(fig_to_png(fig3))

with st.expander("Measured performance of these models (5 seeds, held-out test set)"):
    clean = ckpt.get("multiseed_clean", {})
    rows = []
    for m_name in ("mag", "euv", "concat", "full"):
        if m_name in clean:
            tss = np.array(clean[m_name]["TSS"], dtype=float)
            auc = np.array(clean[m_name]["AUC"], dtype=float)
            rows.append({
                "Model": MODE_LABEL[m_name],
                "TSS": f"{tss.mean():.3f} ± {tss.std():.3f}",
                "AUC": f"{auc.mean():.3f} ± {auc.std():.3f}",
                "Worst seed TSS": f"{tss.min():.3f}",
            })
    if rows:
        st.table(rows)
    st.markdown(
        "Gated fusion beats the best single-input baseline by 0.041 ± 0.013 TSS and "
        "never failed to train, while concatenation collapsed on one seed in five "
        "(TSS 0.080). It does **not** improve robustness to a corrupted modality. "

    )

st.caption(
    f"Checkpoint: {ckpt_path.name} · "
    f"{sum(v.numel() for v in ckpt['state_dicts']['full'].values()):,} parameters per model "
    "· trained on SDOBenchmark (i4Ds/FHNW)."
)
