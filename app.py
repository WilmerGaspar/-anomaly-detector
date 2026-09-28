import streamlit as st

st.set_page_config(page_title="CMS-80", page_icon="\u25a0", layout="wide")
st.markdown("## CMS-80  COSMIC MATERIALS SCOUT")
st.caption("boot liviano · la ciencia se carga al SCAN")

st.sidebar.markdown("`CONFIG`")
source_url = st.sidebar.text_input("URL FITS DIRECTO", value="")
n_null = st.sidebar.slider("NULOS", 10, 80, 24, 2)
use_iso = st.sidebar.checkbox("ISOFOREST", value=True)

tab_scout, tab_cat, tab_rub = st.tabs(["SCOUT", "CATALOGO", "RUBRICA"])

with tab_scout:
    target = st.text_input("Objeto", value="NGC 7023")
    missions = st.multiselect("Misiones", ["HST", "JWST", "ROMAN", "HLSP"], default=["HST", "JWST"])
    if st.button("BUSCAR EN MAST"):
        try:
            from mast_client import search_observations
            with st.spinner("MAST..."):
                st.session_state["mast_obs"] = search_observations(target, missions)
        except Exception as exc:
            st.error(str(exc))
    obs_rows = st.session_state.get("mast_obs") or []
    if obs_rows:
        labels = ["%s | %s | %s | %s" % (r.get("mission"), r.get("instrument"), r.get("filters"), r.get("obs_id")) for r in obs_rows]
        pick = st.selectbox("Observacion", labels)
        chosen = obs_rows[labels.index(pick)]
        if chosen.get("jpeg_url"):
            st.image(chosen["jpeg_url"], caption="preview")
        if st.button("LISTAR FITS"):
            try:
                from mast_client import list_fits_products
                st.session_state["mast_prods"] = list_fits_products(chosen["obsid"])
            except Exception as exc:
                st.error(str(exc))
    prods = st.session_state.get("mast_prods") or []
    if prods:
        shown = [p for p in prods if any(k in p["filename"].lower() for k in ("i2d", "drz", "drc", "x1d"))] or prods
        names = [p["filename"] for p in shown]
        i = st.selectbox("FITS", range(len(names)), format_func=lambda k: names[k])
        if st.button("CARGAR FITS"):
            try:
                from mast_client import download_product
                p = shown[i]
                st.session_state["field_name"] = p["filename"]
                st.session_state["field_bytes"] = download_product(p["uri"], p["filename"])
                st.session_state["started"] = False
            except Exception as exc:
                st.error(str(exc))
    up = st.file_uploader("O carga local", type=["jpg", "jpeg", "png", "tif", "tiff", "fits", "fit"])
    if up is not None:
        st.session_state["field_name"] = up.name
        st.session_state["field_bytes"] = up.getvalue()
        st.session_state["started"] = False
    name = st.session_state.get("field_name")
    raw = st.session_state.get("field_bytes")
    if not raw:
        st.info("Upload un i2d o busca en MAST. Luego INICIAR.")
    else:
        st.success("BUFFER " + str(name))
        if st.button("INICIAR", type="primary"):
            st.session_state["started"] = True
        if st.session_state.get("started"):
            try:
                import io
                import numpy as np
                from PIL import Image
                from astropy.io import fits
                buf = io.BytesIO(raw)
                low = str(name).lower()
                if low.endswith((".fits", ".fit", ".fits.gz")):
                    with fits.open(buf) as hdul:
                        data = None
                        for hdu in hdul:
                            if getattr(hdu, "data", None) is not None:
                                data = hdu.data
                                if str(getattr(hdu, "name", "")) == "SCI":
                                    break
                        if data is None:
                            st.error("NO SCI DATA")
                            st.stop()
                        image = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0)
                        if image.ndim > 2:
                            image = np.squeeze(image)
                            if image.ndim > 2:
                                image = image[0]
                        p2, p98 = np.percentile(image, [2, 98])
                        if p98 > p2:
                            image = np.clip((image - p2) / (p98 - p2), 0, 1)
                        else:
                            image = np.clip(image, 0, 1)
                        header = hdul[0].header
                        meta = {"filename": name, "format": "FITS", "width": int(image.shape[1]), "height": int(image.shape[0]), "is_fits": True, "instrument": str(header.get("INSTRUME", "")), "filter": str(header.get("FILTER", ""))}
                else:
                    im = Image.open(buf).convert("L")
                    image = np.array(im, dtype=np.float32) / 255.0
                    meta = {"filename": name, "format": im.format, "width": im.width, "height": im.height, "is_fits": False, "instrument": "", "filter": ""}
                from provenance import assess_provenance
                prov = assess_provenance(meta["filename"], meta, source_url)
                a, b, c, d = st.columns(4)
                a.metric("FILE", str(name)[:22])
                b.metric("SIZE", "%sx%s" % (meta["width"], meta["height"]))
                c.metric("TRUST", "%.2f" % prov["trust_score"])
                d.metric("GATE", prov["verdict"])
                st.image(image, caption=str(name), use_container_width=True, clamp=True)
            except Exception as exc:
                st.exception(exc)
                st.stop()
            if st.button("SCAN FIELD", type="primary"):
                try:
                    from materials_map import interpret
                    from nos_morphological import morphological_nos
                    from scoring import cheap_descriptor_pvalues, cheap_score_from_image, fdr_decision, surrogate_null_test
                    from candidate_export import build_candidate, dumps_candidate
                    results = {}
                    try:
                        from plugins.kolmogorov_1941 import Kolmogorov1941
                        results["kolmogorov_1941"] = Kolmogorov1941().analyze(image)
                    except Exception as exc:
                        results["kolmogorov_1941"] = {"error": str(exc)}
                    try:
                        from plugins.anisotropy import calculate_anisotropy
                        results["anisotropy"] = calculate_anisotropy(image)
                    except Exception as exc:
                        results["anisotropy"] = {"error": str(exc)}
                    try:
                        from plugins.entropy import calculate_entropy
                        results["entropy"] = calculate_entropy(image)
                    except Exception as exc:
                        results["entropy"] = {"error": str(exc)}
                    try:
                        from plugins.periodicity import analyze_periodicity
                        results["periodicity"] = analyze_periodicity(image)
                    except Exception as exc:
                        results["periodicity"] = {"error": str(exc)}
                    mc = surrogate_null_test(image, cheap_score_from_image(image), cheap_score_from_image, n_simulations=n_null)
                    fdr = fdr_decision(cheap_descriptor_pvalues(image, n_simulations=max(12, n_null // 2)))
                    mc["fdr"] = fdr
                    materials = interpret(results, float(mc.get("z_score") or 0), float(mc.get("p_value") or 1), fdr_pass=bool(fdr.get("fdr_pass")))
                    nos = morphological_nos(results)
                    if prov["verdict"] != "usable_science" or prov.get("product_level") == "detector":
                        materials["is_candidate"] = False
                    st.write(materials.get("verdict"))
                    st.metric("NOS v0", "%.2f" % nos.get("nos", 0))
                    emp = nos.get("empirical") or {}
                    if use_iso and emp.get("available"):
                        st.metric("ISOFOREST", "%.3f" % float(emp.get("nos_isoforest") or 0))
                    k41 = results.get("kolmogorov_1941") or {}
                    if "beta" in k41:
                        st.caption("beta=%.3f se=%s k62=%s" % (k41.get("beta") or 0, k41.get("beta_se"), k41.get("k62_kurtosis")))
                    st.download_button("DUMP JSON", dumps_candidate(build_candidate(meta["filename"], results, materials, prov, nos, mc, meta, source_url)), file_name="cms80_%s.json" % meta["filename"])
                except Exception as exc:
                    st.exception(exc)

with tab_cat:
    st.write("Catalogo curado (sin llamar a MAST al abrir).")
    mission = st.selectbox("Mision", ["JWST", "HST", "ROMAN", "HLSP"])
    curated = {
        "JWST": ["NGC 7023", "M16", "Stephan's Quintet", "SMACS 0723", "Jupiter"],
        "HST": ["M16", "M51", "Crab Nebula", "Hoag Object", "NGC 7023"],
        "ROMAN": ["LMC", "SMC", "Andromeda"],
        "HLSP": ["HUDF", "GOODS-S", "COSMOS"],
    }
    st.write(curated.get(mission, []))
    pick = st.selectbox("Objeto", curated.get(mission, ["NGC 7023"]))
    if st.button("BUSCAR ESTE EN MAST"):
        try:
            from mast_client import search_observations
            st.session_state["mast_obs"] = search_observations(pick, [mission])
            st.success("Listo. Vuelve a SCOUT.")
        except Exception as exc:
            st.error(str(exc))

with tab_rub:
    st.write("Pantalla Oh no = crash de proceso (memoria/import). Este boot no importa matplotlib ni sklearn hasta SCAN.")
    st.table([
        {"Capa": "Ingenieria", "Hoy": "7.0"},
        {"Capa": "Integridad", "Hoy": "7.5"},
        {"Capa": "Descubrimiento", "Hoy": "3.5"},
    ])
