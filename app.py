from __future__ import annotations
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from xr_analyzer import XRNeedAnalyzer, to_pretty_json

APP_TITLE = "XR NeedStruct — Extracteur de besoins XR industriels"
DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="XR NeedStruct", page_icon="🧩", layout="wide")

@st.cache_resource
def load_analyzer():
    return XRNeedAnalyzer(DATA_DIR)

@st.cache_data
def load_examples():
    path = DATA_DIR / "examples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from xr_data import FILES
    return json.loads(FILES.get("examples.json", "[]"))

@st.cache_data
def load_csv(name: str):
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path)
    from io import StringIO
    from xr_data import FILES
    return pd.read_csv(StringIO(FILES[name]))


def card(label: str, value: str | int | float | None, help_text: str = ""):
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"### {value if value not in [None, ''] else '—'}")
        if help_text:
            st.caption(help_text)


def show_list(title: str, items, key="nom"):
    st.markdown(f"#### {title}")
    if not items:
        st.info("Aucun élément détecté.")
        return
    for item in items:
        if isinstance(item, dict):
            name = item.get(key) or item.get("nom") or item.get("categorie") or str(item)
            with st.expander(str(name), expanded=False):
                st.json(item, expanded=True)
        else:
            st.write("-", item)


st.title(APP_TITLE)
st.caption("Transforme un texte libre utilisateur en fiche de besoin XR structurée : contexte, activités, hardware, objets virtuels et environnement réel.")

analyzer = load_analyzer()
examples = load_examples()

with st.sidebar:
    st.header("Paramètres")
    threshold = st.slider("Seuil activités", min_value=0.2, max_value=2.0, value=0.55, step=0.05,
                          help="Plus le seuil est élevé, moins l'application détecte d'activités faibles ou ambiguës.")
    show_debug = st.checkbox("Afficher le debug", value=False)
    st.divider()
    st.subheader("Exemples")
    selected_example = st.selectbox("Charger un exemple", ["—"] + [ex["name"] for ex in examples])

initial_text = ""
if selected_example != "—":
    initial_text = next(ex["text"] for ex in examples if ex["name"] == selected_example)

text = st.text_area(
    "Texte utilisateur",
    value=initial_text,
    height=170,
    placeholder="Exemple : Je veux une application AR pour guider un technicien pendant la maintenance d’une pompe réelle...",
)

col_a, col_b = st.columns([1, 4])
with col_a:
    analyze_btn = st.button("Analyser", type="primary", use_container_width=True)
with col_b:
    st.caption("Le moteur est local et déterministe : il utilise des datasets CSV, des synonymes, des règles contextuelles et une gestion des objets inconnus.")

if analyze_btn or text.strip():
    if not text.strip():
        st.warning("Écris un texte utilisateur à analyser.")
        st.stop()
    result = analyzer.analyze(text, activity_threshold=threshold)
    context = result["contexte_utilisation"]
    xr = context["type_XR"]
    domain = context["domaine_application"]
    activities = context["activites"]
    hw = result["environnement_immersif"]
    scene = result["scene_et_objets_virtuels"]
    real = result["environnement_reel"]

    st.divider()
    st.subheader("Résumé")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Technologie XR", xr.get("valeur"), f"Confiance : {xr.get('confiance', '—')}")
    with c2:
        card("Domaine", domain.get("valeur"), f"Confiance : {domain.get('confiance', '—')}")
    with c3:
        card("Activités détectées", len(activities), ", ".join([a["nom"] for a in activities[:3]]) + ("..." if len(activities) > 3 else ""))
    with c4:
        card("Interaction avec le réel", "Oui" if real.get("interaction_avec_reel") else "Non", real.get("lieu_reel") or "")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Contexte", "Hardware", "Objets virtuels", "Environnement réel", "JSON", "Datasets"])

    with tab1:
        left, right = st.columns(2)
        with left:
            st.markdown("### Technologie XR")
            st.json(xr)
            st.markdown("### Domaine d'application")
            st.json(domain)
        with right:
            st.markdown("### Activités")
            if activities:
                st.dataframe(pd.DataFrame(activities), use_container_width=True, hide_index=True)
            else:
                st.info("Aucune activité détectée avec le seuil actuel.")

    with tab2:
        st.markdown("### Matériel détecté ou recommandé")
        col1, col2 = st.columns(2)
        with col1:
            show_list("Hardware explicitement mentionné", hw.get("hardware_explicitement_mentionne", []))
        with col2:
            show_list("Hardware suggéré", hw.get("hardware_suggere", []))
        st.markdown("#### Capteurs / capacités nécessaires")
        st.write(hw.get("capteurs_necessaires", []) or "Aucun capteur supplémentaire proposé.")

    with tab3:
        st.markdown("### Scène principale")
        st.json(scene.get("scene_principale", {}))
        c1, c2 = st.columns(2)
        with c1:
            show_list("Objets 3D principaux", scene.get("objets_3D_principaux", []))
            show_list("Sous-composants", scene.get("sous_composants", []))
        with c2:
            show_list("Petits composants", scene.get("petits_composants", []))
            show_list("Outils", scene.get("outils", []))
        show_list("Éléments virtuels XR", scene.get("elements_virtuels_XR", []))
        show_list("Objets inconnus à confirmer", scene.get("objets_inconnus", []))

    with tab4:
        st.markdown("### Environnement réel")
        st.json({k: v for k, v in real.items() if k not in ["objets_reels", "relations_reel_virtuel"]})
        show_list("Objets réels", real.get("objets_reels", []))
        show_list("Relations réel-virtuel", real.get("relations_reel_virtuel", []), key="objet")

    with tab5:
        st.markdown("### Sortie JSON structurée")
        json_text = to_pretty_json({k: v for k, v in result.items() if k != "debug"})
        st.code(json_text, language="json")
        st.download_button("Télécharger le JSON", json_text, file_name="besoin_xr_structure.json", mime="application/json")

    with tab6:
        st.markdown("### Bases utilisées par le moteur")
        dataset_name = st.selectbox("Dataset", [
            "xr_domain_keywords.csv", "activity_keywords.csv", "hardware_keywords.csv", "hardware_inference_rules.csv",
            "known_objects.csv", "object_families.csv", "object_action_rules.csv", "real_environment_indicators.csv",
            "description_templates.csv",
        ])
        df = load_csv(dataset_name)
        st.dataframe(df, use_container_width=True)

    if show_debug:
        st.divider()
        st.subheader("Debug")
        st.json(result.get("debug", {}))
else:
    st.info("Entre un texte utilisateur ou charge un exemple, puis clique sur Analyser.")

st.divider()
st.caption("Version finale — pipeline hybride : scoring lexical, synonymes multilingues, règles contextuelles, recommandation hardware, extraction d'objets connus/inconnus et relation réel-virtuel.")
