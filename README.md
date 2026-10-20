# XR NeedStruct — Streamlit

Projet Streamlit complet pour transformer un texte utilisateur en fiche structurée de besoin XR industriel.

## Fonctionnalités

- Détection du contexte d'utilisation : **VR / AR / MR**, **Conception / Production / Maintenance**.
- Détection multi-label des activités XR : formation, simulation, assistance, guidage, assemblage, inspection, diagnostic, etc.
- Capture du hardware mentionné : casque VR, HoloLens, tablette, caméra, marqueurs, capteurs...
- Recommandation du hardware si absent, selon **technologie XR + domaine + activités**.
- Extraction des scènes et objets virtuels : scène principale, objets 3D, sous-composants, petits composants, outils, éléments XR.
- Détection de l'environnement réel : objets physiques, interaction AR/MR, relation réel-virtuel.
- Gestion des synonymes, variantes français/anglais/franglais et fautes fréquentes.
- Gestion des objets inconnus : ils ne sont pas ignorés, ils sont conservés comme objets à confirmer.
- Export JSON.

## Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```text
xr_needstruct_streamlit_v2/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── xr_domain_keywords.csv
│   ├── activity_keywords.csv
│   ├── hardware_keywords.csv
│   ├── hardware_inference_rules.csv
│   ├── hardware_keywords.csv
│   ├── object_families.csv
│   ├── known_objects.csv
│   ├── object_action_rules.csv
│   ├── real_environment_indicators.csv
│   ├── description_templates.csv
│   └── examples.json
└── utils/
    ├── __init__.py
    └── analyzer.py
```

## Méthode

Le pipeline combine :

1. Normalisation du texte.
2. Recherche des mots-clés et synonymes.
3. Calcul des scores XR et domaines.
4. Détection multi-label des activités.
5. Détection ou recommandation hardware.
6. Extraction des objets connus et inconnus.
7. Détection de l'interaction réelle.
8. Génération de la sortie JSON.

## Sortie principale

```json
{
  "contexte_utilisation": {},
  "environnement_immersif": {},
  "scene_et_objets_virtuels": {},
  "environnement_reel": {}
}
```

## Personnalisation

Tu peux enrichir les fichiers CSV dans `data/` sans modifier le code :

- Ajouter des mots-clés XR/domaines dans `xr_domain_keywords.csv`.
- Ajouter des activités dans `activity_keywords.csv`.
- Ajouter du hardware dans `hardware_keywords.csv`.
- Ajouter des objets industriels dans `known_objects.csv`.
- Ajouter des règles de recommandation dans `hardware_inference_rules.csv`.


## Déploiement en ligne (Streamlit Community Cloud)

1. Pousser ce dossier à la racine d'un dépôt GitHub (`app.py` à la racine).
2. Sur https://share.streamlit.io : « Create app », choisir le dépôt, la branche `main` et le fichier `app.py`.
3. Après installation, l'application est accessible via une URL publique (ou restreinte à des e-mails autorisés dans les réglages de partage).
