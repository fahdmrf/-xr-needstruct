from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover
    fuzz = None
    process = None

DATA_DIR = Path(__file__).resolve().parent / "data"

TECHS = ["VR", "AR", "MR"]
DOMAINS = ["Conception", "Production", "Maintenance"]
ACTIVITY_LIST = [
    "Visualisation", "Formation", "Simulation", "Assistance", "Guidage procédural",
    "Assemblage", "Démontage", "Contrôle qualité", "Inspection", "Diagnostic", "Mesure",
    "Calibration", "Réparation", "Remplacement", "Prototypage virtuel", "Revue collaborative",
    "Évaluation ergonomique", "Planification d’implantation", "Optimisation des flux", "Navigation",
    "Localisation et tracking", "Logistique", "Capture de mouvement", "Analyse et aide à la décision",
]

ACTION_WORDS = [
    "assembler", "monter", "fixer", "installer", "visser", "serrer", "connecter", "brancher",
    "demonter", "retirer", "ouvrir", "devisser", "desserrer", "enlever", "deconnecter", "debrancher",
    "remplacer", "changer", "reparer", "corriger", "inspecter", "verifier", "controler", "diagnostiquer",
    "mesurer", "scanner", "detecter", "reconnaitre", "localiser", "calibrer", "regler", "aligner", "superposer",
    "serrage", "tighten", "tightening",
]

STOP_NOUNS = set("""
application appli systeme outil module besoin utilisateur technicien operateur equipe texte projet objectif solution usage cas etape procedure aide support
realite virtuelle augmentee mixte ar vr mr industrie industriel industrielle donnees sortie entree resultat exemple chose version
pour avec dans sur pendant doit doivent je tu il elle nous vous ils elles le la les un une des du de d au aux a et ou qui que quoi comment faire veux veut souhaite souhaitons aider voir montrer afficher demontage montage maintenance formation assistance simulation
""".split())


def normalize_text(text: str) -> str:
    """Normalize French/English industrial text for robust keyword matching."""
    if text is None:
        return ""
    text = str(text).lower().replace("’", "'").replace("œ", "oe")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9+/#\-\s']+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # simple typo normalization for frequent industrial terms
    replacements = {
        "goulon": "boulon", "goulons": "boulons", "boulonns": "boulons", "viss": "vis",
        "ecrout": "ecrou", "ecrouts": "ecrous", "cle alen": "cle allen", "tourne vise": "tournevis",
        "maintenence": "maintenance", "realitee": "realite", "augmantee": "augmentee",
        "macine": "machine", "machien": "machine", "machinne": "machine",
    }
    for wrong, right in replacements.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    return text


def split_cell(value: Any) -> List[str]:
    if pd.isna(value) or value is None:
        return []
    return [x.strip() for x in re.split(r"[;|,]", str(value)) if x.strip()]


@dataclass
class Match:
    keyword: str
    normalized_keyword: str
    value: str
    weight: float
    confidence: str
    ambiguous: bool
    start: int = -1
    end: int = -1
    source: str = "keyword"


class XRNeedAnalyzer:
    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.xr_domain = self._read_csv("xr_domain_keywords.csv")
        self.activities = self._read_csv("activity_keywords.csv")
        self.hardware = self._read_csv("hardware_keywords.csv")
        self.hw_rules = self._read_csv("hardware_inference_rules.csv")
        self.objects = self._read_csv("known_objects.csv")
        self.real_indicators = self._read_csv("real_environment_indicators.csv")
        self.action_rules = self._read_csv("object_action_rules.csv")
        self.templates = self._read_csv("description_templates.csv")
        self._prepare_tables()

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = self.data_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path).fillna("")

    def _prepare_tables(self):
        for attr in ["xr_domain", "activities", "hardware", "objects", "real_indicators"]:
            df = getattr(self, attr)
            if not df.empty and "keyword" in df.columns:
                df["normalized_keyword"] = df["keyword"].astype(str).apply(normalize_text)
                setattr(self, attr, df)
        # Expand known object synonyms as matchable aliases
        if not self.objects.empty:
            rows = []
            for _, row in self.objects.iterrows():
                base = row.to_dict()
                rows.append(base)
                for syn in split_cell(row.get("synonyms", "")):
                    r = base.copy()
                    r["keyword"] = syn
                    r["normalized_keyword"] = normalize_text(syn)
                    r["alias_of"] = base.get("keyword", "")
                    rows.append(r)
            self.object_aliases = pd.DataFrame(rows).fillna("")
        else:
            self.object_aliases = pd.DataFrame()

    def _contains_phrase(self, text_norm: str, kw_norm: str) -> Tuple[bool, int, int]:
        if not kw_norm:
            return False, -1, -1
        # Boundary-aware phrase match, allowing +/# terms.
        pattern = r"(?<![a-z0-9])" + re.escape(kw_norm) + r"(?![a-z0-9])"
        m = re.search(pattern, text_norm)
        if m:
            return True, m.start(), m.end()
        return False, -1, -1

    def _match_rows(self, text_norm: str, df: pd.DataFrame, value_col: str, weight_col: str = "weight") -> List[Match]:
        matches: List[Match] = []
        if df.empty:
            return matches
        for _, row in df.iterrows():
            kw = row.get("keyword", "")
            kw_norm = row.get("normalized_keyword", normalize_text(kw))
            found, start, end = self._contains_phrase(text_norm, kw_norm)
            if found:
                weight = row.get(weight_col, 1.0)
                try:
                    weight = float(weight)
                except Exception:
                    weight = 1.0
                matches.append(Match(
                    keyword=str(kw), normalized_keyword=kw_norm, value=str(row.get(value_col, "")), weight=weight,
                    confidence=str(row.get("confidence", "medium")), ambiguous=bool(row.get("ambiguous", False)),
                    start=start, end=end,
                ))
        # keep longest match first to reduce duplicates later
        matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
        return matches

    def classify_context(self, text_norm: str) -> Dict[str, Any]:
        scores_xr = {t: 0.0 for t in TECHS}
        scores_domain = {d: 0.0 for d in DOMAINS}
        evidence_xr = {t: [] for t in TECHS}
        evidence_domain = {d: [] for d in DOMAINS}
        if self.xr_domain.empty:
            return {}
        for _, row in self.xr_domain.iterrows():
            kw = row.get("keyword", "")
            kw_norm = row.get("normalized_keyword", normalize_text(kw))
            found, _, _ = self._contains_phrase(text_norm, kw_norm)
            if not found:
                continue
            for tech, col in [("VR", "vr_weight"), ("AR", "ar_weight"), ("MR", "mr_weight")]:
                w = float(row.get(col, 0) or 0)
                if w:
                    scores_xr[tech] += w
                    evidence_xr[tech].append(str(kw))
            for dom, col in [("Production", "production_weight"), ("Conception", "conception_weight"), ("Maintenance", "maintenance_weight")]:
                w = float(row.get(col, 0) or 0)
                if w:
                    scores_domain[dom] += w
                    evidence_domain[dom].append(str(kw))
        xr_val, xr_conf = self._winner(scores_xr)
        dom_val, dom_conf = self._winner(scores_domain)
        return {
            "type_XR": {"valeur": xr_val, "source": "explicite/inférée" if xr_val else "absente", "score": round(scores_xr.get(xr_val, 0), 3) if xr_val else 0, "confiance": xr_conf, "evidence": evidence_xr.get(xr_val, []) if xr_val else []},
            "domaine_application": {"valeur": dom_val, "source": "explicite/inférée" if dom_val else "absente", "score": round(scores_domain.get(dom_val, 0), 3) if dom_val else 0, "confiance": dom_conf, "evidence": evidence_domain.get(dom_val, []) if dom_val else []},
            "scores": {"XR": scores_xr, "domaines": scores_domain},
        }

    def _winner(self, scores: Dict[str, float]) -> Tuple[str | None, str]:
        if not scores or max(scores.values()) <= 0:
            return None, "low"
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 0
        if best_score >= 2.0 and best_score >= second_score * 1.35:
            return best, "high"
        if best_score >= 0.8 and best_score >= second_score * 1.1:
            return best, "medium"
        return best, "low/ambiguous"

    def detect_activities(self, text_norm: str, threshold: float = 0.55) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {a: 0.0 for a in ACTIVITY_LIST}
        evidence: Dict[str, List[str]] = {a: [] for a in ACTIVITY_LIST}
        ambiguous_hits = []
        matches = self._match_rows(text_norm, self.activities, "activity", "weight")
        for m in matches:
            if not m.value:
                continue
            boost = self._context_boost(text_norm, m.value, m.normalized_keyword)
            scores[m.value] = scores.get(m.value, 0.0) + m.weight * boost
            evidence.setdefault(m.value, []).append(m.keyword)
            if m.ambiguous:
                ambiguous_hits.append(m.keyword)
        # Useful co-occurrence rules
        if self._has_any(text_norm, ["conformite", "qualite", "ok nok", "ok/nok"]) and self._has_any(text_norm, ["verifier", "controler", "valider"]):
            scores["Contrôle qualité"] += 0.8; evidence["Contrôle qualité"].append("règle: vérifier + conformité/qualité")
        if self._has_any(text_norm, ["dimension", "cote", "tolerance", "couple", "torque", "pression", "temperature", "vibration"]):
            scores["Mesure"] += 0.6; evidence["Mesure"].append("règle: grandeur mesurable")
        if self._has_any(text_norm, ["panne", "anomalie", "dysfonctionnement", "defaillance"]):
            scores["Diagnostic"] += 0.7; evidence["Diagnostic"].append("règle: panne/anomalie")
        if self._has_any(text_norm, ["etape", "procedure", "instruction", "checklist", "pas a pas"]):
            scores["Guidage procédural"] += 0.6; evidence["Guidage procédural"].append("règle: étapes/procédure")
        if self._has_any(text_norm, ["serrage", "serrer", "visser", "tightening", "tighten"]) and self._has_any(text_norm, ["boulon", "boulons", "vis", "ecrou", "ecrous", "fixation"]):
            scores["Assemblage"] += 0.9; evidence["Assemblage"].append("règle: serrage + fixation")
            scores["Mesure"] += 0.35; evidence["Mesure"].append("règle: serrage/couple potentiel")
        out = []
        for act, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                out.append({
                    "nom": act,
                    "score": round(score, 3),
                    "confiance": "high" if score >= 1.4 else "medium" if score >= 0.8 else "low",
                    "termes_detectes": list(dict.fromkeys(evidence.get(act, [])))[:10]
                })
        return out[:8]

    def _context_boost(self, text_norm: str, activity: str, kw: str) -> float:
        context_map = {
            "Contrôle qualité": ["conformite", "qualite", "defaut", "ok nok"],
            "Mesure": ["dimension", "tolerance", "cote", "couple", "pression", "temperature"],
            "Inspection": ["etat", "visuel", "usure", "machine", "piece"],
            "Navigation": ["vers", "chemin", "parcours", "zone", "itineraire"],
            "Guidage procédural": ["etape", "procedure", "instruction", "checklist"],
            "Diagnostic": ["panne", "anomalie", "defaillance", "cause"],
            "Localisation et tracking": ["position", "mouvement", "emplacement", "suivi"],
        }
        positives = context_map.get(activity, [])
        if positives and any(self._contains_phrase(text_norm, p)[0] for p in positives):
            return 1.35
        return 1.0

    def _has_any(self, text_norm: str, terms: List[str]) -> bool:
        return any(self._contains_phrase(text_norm, normalize_text(t))[0] for t in terms)

    def _infer_context_if_missing(self, text_norm: str, context: Dict[str, Any], activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Infer XR/domain for short, incomplete user phrases.
        The goal is not to force a result, but to avoid returning empty context when
        the text contains strong operational clues such as "simuler", "serrage",
        "boulons" or "machine CNC".
        """
        act_names = [a.get("nom", "") for a in activities]
        xr = context.get("type_XR", {}).get("valeur")
        dom = context.get("domaine_application", {}).get("valeur")
        if not xr:
            if "Simulation" in act_names or self._has_any(text_norm, ["simuler", "simulation", "virtuel", "environnement simule"]):
                context["type_XR"] = {
                    "valeur": "VR",
                    "source": "inférée depuis l’activité de simulation",
                    "score": 0.75,
                    "confiance": "medium",
                    "evidence": ["simulation/simuler"]
                }
            elif self._has_any(text_norm, ["sur la machine", "sur la piece", "sur le moteur", "sur la pompe", "afficher sur", "superposer", "overlay"]):
                context["type_XR"] = {
                    "valeur": "AR",
                    "source": "inférée depuis l’affichage sur un objet industriel",
                    "score": 0.7,
                    "confiance": "medium",
                    "evidence": ["sur un objet industriel"]
                }
        if not dom:
            maintenance_clues = ["reparer", "maintenance", "panne", "diagnostic", "inspecter", "remplacer", "demonter", "devisser", "retirer"]
            production_clues = ["assembler", "montage", "serrage", "serrer", "visser", "boulon", "boulons", "vis", "ecrou", "machine cnc", "ligne de production"]
            conception_clues = ["prototype", "cao", "maquette", "revue", "ergonomie", "conception", "design"]
            if self._has_any(text_norm, maintenance_clues):
                context["domaine_application"] = {
                    "valeur": "Maintenance", "source": "inférée depuis les actions techniques", "score": 0.75,
                    "confiance": "medium", "evidence": ["action de maintenance"]
                }
            elif self._has_any(text_norm, production_clues) or any(a in act_names for a in ["Assemblage", "Contrôle qualité", "Mesure"]):
                context["domaine_application"] = {
                    "valeur": "Production", "source": "inférée depuis l’assemblage/serrage et les objets industriels", "score": 0.75,
                    "confiance": "medium", "evidence": ["serrage/assemblage", "machine ou fixation"]
                }
            elif self._has_any(text_norm, conception_clues):
                context["domaine_application"] = {
                    "valeur": "Conception", "source": "inférée depuis les indices de conception", "score": 0.75,
                    "confiance": "medium", "evidence": ["prototype/conception"]
                }
        return context

    def detect_hardware(self, text_norm: str, xr_value: str | None, domain_value: str | None, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        detected = []
        matches = self._match_rows(text_norm, self.hardware, "hardware_category", "vr_weight")
        # Use explicit matches, not xr-specific weight; compute confidence from table
        seen = set()
        for _, row in self.hardware.iterrows():
            kw = row.get("keyword", "")
            kw_norm = row.get("normalized_keyword", normalize_text(kw))
            found, _, _ = self._contains_phrase(text_norm, kw_norm)
            if not found: continue
            cat = str(row.get("hardware_category", kw))
            if cat in seen: continue
            seen.add(cat)
            detected.append({"nom": str(kw), "categorie": cat, "source": "explicite", "confiance": str(row.get("confidence", "medium"))})
        suggested = []
        sensors = []
        # If no strong hardware mentioned, recommend from rules.
        act_names = [a["nom"] for a in activities]
        if self.hw_rules.empty:
            return {"hardware_explicitement_mentionne": detected, "hardware_suggere": [], "capteurs_necessaires": []}
        for _, rule in self.hw_rules.iterrows():
            xr_ok = rule.get("xr", "Any") in ["Any", xr_value]
            dom_ok = rule.get("domain", "Any") in ["Any", domain_value]
            act_ok = rule.get("activity", "Any") == "Any" or rule.get("activity") in act_names
            if xr_ok and dom_ok and act_ok:
                for item in split_cell(rule.get("recommended_hardware", "")):
                    suggested.append({"nom": item, "categorie": self._hardware_category(item), "raison": str(rule.get("reason", "")), "confiance": "high" if xr_value != "Any" else "medium"})
                sensors.extend(split_cell(rule.get("required_sensors", "")))
        if detected:
            # keep recommendations only for complementary sensors/capabilities
            explicit_cats = {x["categorie"] for x in detected}
            suggested = [x for x in suggested if x["categorie"] not in explicit_cats and x["nom"] not in explicit_cats]
        return {
            "hardware_explicitement_mentionne": detected,
            "hardware_suggere": self._dedup_dicts(suggested, "nom")[:6],
            "capteurs_necessaires": list(dict.fromkeys(sensors))[:8],
        }

    def _hardware_category(self, name: str) -> str:
        n = normalize_text(name)
        for _, row in self.hardware.iterrows():
            if normalize_text(str(row.get("hardware_category", ""))) == n or normalize_text(str(row.get("keyword", ""))) == n:
                return str(row.get("hardware_category", name))
        return name

    def extract_scene_and_objects(self, text: str, text_norm: str, domain: str | None, activities: List[Dict[str, Any]]) -> Dict[str, Any]:
        object_matches = self._match_objects(text_norm)
        actions = self._extract_actions(text_norm)
        act_names = [a["nom"] for a in activities]
        # Unknown technical noun chunks
        unknowns = self._extract_unknown_objects(text_norm, object_matches)
        scene_candidates = [o for o in object_matches if o.get("object_level") == "scene"]
        scene = self._build_scene(scene_candidates, domain, act_names)
        principal, sub, small, tools, virtual_elements = [], [], [], [], []
        for o in object_matches:
            level = o.get("object_level", "")
            item = self._object_to_output(o, actions, domain, act_names, text_norm)
            if level == "scene":
                continue
            if level == "objet principal": principal.append(item)
            elif level == "sous-composant": sub.append(item)
            elif level == "petit composant": small.append(item)
            elif level == "outil": tools.append(item)
            elif level == "élément virtuel": virtual_elements.append(item)
        # Promote unknowns near actions to principal objects
        for u in unknowns:
            if u["categorie_probable"] in ["équipement industriel probable", "objet technique probable"]:
                principal.append({
                    "nom": u["nom"], "categorie": u["categorie_probable"],
                    "description": "Objet technique identifié dans le texte utilisateur, à confirmer ou à enrichir dans la base.",
                    "role": "objet potentiel du scénario XR", "actions_associees": u.get("actions_associees", []),
                    "presence": "virtuel ou réel+virtuel selon le type XR", "niveau_detail_3D": "élevé" if u.get("actions_associees") else "moyen",
                    "source": "extraction automatique", "confiance": "medium", "a_confirmer": True
                })
        return {
            "scene_principale": scene,
            "objets_3D_principaux": self._dedup_dicts(principal, "nom")[:8],
            "sous_composants": self._dedup_dicts(sub, "nom")[:12],
            "petits_composants": self._dedup_dicts(small, "nom")[:12],
            "outils": self._dedup_dicts(tools, "nom")[:8],
            "elements_virtuels_XR": self._dedup_dicts(virtual_elements, "nom")[:8],
            "objets_inconnus": unknowns[:8],
        }

    def _match_objects(self, text_norm: str) -> List[Dict[str, Any]]:
        matches = []
        for _, row in self.object_aliases.iterrows():
            kw_norm = row.get("normalized_keyword", normalize_text(row.get("keyword", "")))
            found, start, end = self._contains_phrase(text_norm, kw_norm)
            if found:
                item = row.to_dict()
                item["matched_keyword"] = row.get("keyword", "")
                item["canonical_name"] = row.get("alias_of", "") or row.get("keyword", "")
                item["start"] = start; item["end"] = end
                matches.append(item)
        # Prefer longer spans first; remove duplicated and nested shorter matches.
        matches.sort(key=lambda o: (-(o["end"] - o["start"]), o["start"]))
        out = []
        seen_names = set()
        occupied_spans = []
        for o in matches:
            name = normalize_text(str(o.get("canonical_name", "")))
            if name in seen_names:
                continue
            s, e = int(o.get("start", -1)), int(o.get("end", -1))
            nested = any(s >= os and e <= oe for os, oe in occupied_spans)
            if nested:
                continue
            out.append(o)
            seen_names.add(name)
            occupied_spans.append((s, e))
        out.sort(key=lambda o: o.get("start", 0))
        return out

    def _extract_actions(self, text_norm: str) -> List[str]:
        actions = []
        for a in ACTION_WORDS:
            if self._contains_phrase(text_norm, a)[0]:
                actions.append(a)
        return list(dict.fromkeys(actions))

    def _extract_unknown_objects(self, text_norm: str, known_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Use all known aliases to avoid re-emitting partial known objects as unknown.
        known_norms = {normalize_text(str(o.get("canonical_name", ""))) for o in known_matches}
        if not self.object_aliases.empty:
            known_norms |= {normalize_text(str(x)) for x in self.object_aliases.get("normalized_keyword", []) if str(x)}
        activity_norms = set()
        if not self.activities.empty:
            activity_norms = {normalize_text(str(x)) for x in self.activities.get("normalized_keyword", []) if str(x)}
        candidates = []
        # noun chunks after determiners; intentionally simple and explainable
        patterns = [
            r"\b(?:un|une|le|la|les|des|du|de la|de l|d un|d une)\s+([a-z0-9][a-z0-9\-']+(?:\s+[a-z0-9][a-z0-9\-']+){0,2})",
            r"\b(?:sur|dans|avec|pour|de|du|des)\s+(?:la|le|les|un|une|des)?\s*([a-z0-9][a-z0-9\-']+(?:\s+[a-z0-9][a-z0-9\-']+){0,2})",
        ]
        for pat in patterns:
            for m in re.finditer(pat, text_norm):
                chunk = m.group(1).strip()
                words = chunk.split()
                # truncate at the first stop/action/activity word inside the candidate
                cleaned = []
                for w in words:
                    if w in STOP_NOUNS or w in ACTION_WORDS or w in activity_norms:
                        break
                    cleaned.append(w)
                words = cleaned
                while words and words[-1] in STOP_NOUNS:
                    words.pop()
                while words and words[0] in STOP_NOUNS:
                    words.pop(0)
                if not words: continue
                chunk = " ".join(words)
                if len(chunk) < 3 or chunk in known_norms or chunk in STOP_NOUNS:
                    continue
                chunk_words = chunk.split()
                if any(w in STOP_NOUNS for w in chunk_words):
                    continue
                if any(w in ACTION_WORDS for w in chunk_words):
                    continue
                if chunk in activity_norms:
                    continue
                # avoid chunks containing known objects, e.g. "pompe reelle" when "pompe" is already known
                if any((kn and (self._contains_phrase(chunk, kn)[0] or self._contains_phrase(kn, chunk)[0])) for kn in known_norms):
                    continue
                # reject chunks that are mostly activities/hardware/general
                if any(self._contains_phrase(normalize_text(chunk), normalize_text(x))[0] for x in ["application", "realite", "technicien", "operateur", "instruction", "etape", "maintenance", "production", "conception"]):
                    continue
                near_action = [a for a in ACTION_WORDS if a in text_norm[max(0, m.start()-45): min(len(text_norm), m.end()+45)]]
                categorie = "équipement industriel probable" if near_action else "objet technique probable"
                candidates.append({"nom": chunk, "categorie_probable": categorie, "raison": "groupe nominal proche d’une action technique" if near_action else "groupe nominal détecté", "actions_associees": near_action[:3], "a_confirmer": True})
        return self._dedup_dicts(candidates, "nom")

    def _build_scene(self, scene_candidates: List[Dict[str, Any]], domain: str | None, acts: List[str]) -> Dict[str, Any]:
        if scene_candidates:
            name = str(scene_candidates[0].get("canonical_name", scene_candidates[0].get("keyword", ""))).title()
            return {"nom": name, "description": str(scene_candidates[0].get("default_description", "Environnement principal du scénario XR.")), "source": "explicite", "confiance": "high"}
        if domain == "Maintenance":
            return {"nom": "Zone de maintenance industrielle", "description": "Scène XR centrée sur une intervention technique en environnement industriel.", "source": "inférée depuis le domaine Maintenance", "confiance": "medium"}
        if domain == "Production":
            return {"nom": "Environnement de production", "description": "Scène XR centrée sur un poste, une ligne ou une opération de production.", "source": "inférée depuis le domaine Production", "confiance": "medium"}
        if domain == "Conception":
            return {"nom": "Espace de conception ou de revue", "description": "Scène XR dédiée à la visualisation, la revue ou la validation d’un produit ou prototype.", "source": "inférée depuis le domaine Conception", "confiance": "medium"}
        return {"nom": None, "description": None, "source": "absente", "confiance": "low"}

    def _object_to_output(self, o: Dict[str, Any], actions: List[str], domain: str | None, acts: List[str], text_norm: str) -> Dict[str, Any]:
        name = str(o.get("canonical_name", o.get("keyword", ""))).title()
        level = str(o.get("object_level", ""))
        nearby_actions = self._near_actions(text_norm, int(o.get("start", 0)), int(o.get("end", 0))) or actions[:2]
        lod = self._lod_for_object(level, nearby_actions, acts)
        desc = str(o.get("default_description", "Objet identifié dans le texte utilisateur."))
        role = self._role_for_object(level, nearby_actions, acts)
        presence = "virtuel"
        if self._has_any(text_norm, ["reel", "reelle", "physique", "scanner", "superposer", "aligner", "ancrer"]):
            presence = "réel + virtuel probable"
        return {
            "nom": name,
            "categorie": str(o.get("family", "objet")),
            "description": desc,
            "role": role,
            "actions_associees": nearby_actions,
            "presence": presence,
            "niveau_detail_3D": lod,
            "source": "base de données" if not o.get("alias_of") else f"synonyme de {o.get('alias_of')}",
            "confiance": str(o.get("confidence", "medium")),
        }

    def _near_actions(self, text_norm: str, start: int, end: int, window: int = 55) -> List[str]:
        ctx = text_norm[max(0, start-window): min(len(text_norm), end+window)]
        return [a for a in ACTION_WORDS if self._contains_phrase(ctx, a)[0]][:5]

    def _lod_for_object(self, level: str, actions: List[str], activities: List[str]) -> str:
        action_set = set(actions)
        if action_set & {"mesurer", "inspecter", "calibrer", "diagnostiquer", "scanner", "detecter", "reconnaitre"}:
            return "très élevé"
        if action_set & {"retirer", "devisser", "desserrer", "serrer", "serrage", "visser", "remplacer", "assembler", "demonter", "connecter", "deconnecter", "tighten", "tightening"}:
            return "élevé"
        if level in ["petit composant", "sous-composant"] and any(a in activities for a in ["Assemblage", "Démontage", "Réparation", "Remplacement"]):
            return "élevé"
        if level == "scene": return "moyen"
        return "moyen"

    def _role_for_object(self, level: str, actions: List[str], activities: List[str]) -> str:
        if level == "objet principal": return "objet central du scénario XR"
        if level == "sous-composant": return "élément associé à l’objet principal"
        if level == "petit composant": return "élément de détail potentiellement manipulé ou guidé"
        if level == "outil": return "outil utilisé pour réaliser une action technique"
        if level == "élément virtuel": return "aide visuelle ou information XR"
        return "élément du scénario XR"

    def extract_real_environment(self, text_norm: str, xr_value: str | None, scene_objects: Dict[str, Any]) -> Dict[str, Any]:
        indicators = []
        types = []
        for _, row in self.real_indicators.iterrows():
            kw = row.get("keyword", "")
            kw_norm = row.get("normalized_keyword", normalize_text(kw))
            if self._contains_phrase(text_norm, kw_norm)[0]:
                indicators.append(str(kw)); types.append(str(row.get("interaction_type", "interaction réelle")))
        # In short phrases, "sur la machine" can simply mean that the virtual action is performed
        # on the virtual machine. For VR, we only activate the real environment if the text contains
        # explicit real-world cues.
        explicit_real = self._has_any(text_norm, ["reel", "reelle", "physique", "sur site", "terrain", "dans l atelier", "dans l usine", "scanner", "scan", "reconnaitre", "identifier", "aligner", "ancrer"])
        interaction = bool(indicators) or xr_value in ["AR", "MR"] and bool(scene_objects.get("objets_3D_principaux") or scene_objects.get("sous_composants") or scene_objects.get("petits_composants"))
        if xr_value == "VR" and not explicit_real:
            return {"interaction_avec_reel": False, "lieu_reel": None, "indices_detection": [], "type_interaction": [], "objets_reels": [], "relations_reel_virtuel": [], "commentaire": "Le scénario semble entièrement virtuel ou simulé ; aucun objet réel n’est explicitement mentionné."}
        real_objs = []
        if interaction:
            for group in ["objets_3D_principaux", "sous_composants", "petits_composants"]:
                for obj in scene_objects.get(group, []):
                    if xr_value in ["AR", "MR"] or "réel" in obj.get("presence", "") or indicators:
                        real_objs.append({
                            "nom": f"{obj['nom']} réel" if not obj["nom"].lower().endswith("réel") else obj["nom"],
                            "categorie": obj.get("categorie", "objet réel"),
                            "description": self._real_description(obj, xr_value, types),
                            "role": "objet réel principal" if group == "objets_3D_principaux" else "objet réel associé à l’interaction",
                            "actions_associees": obj.get("actions_associees", []),
                            "interaction_XR": self._interaction_label(xr_value, types),
                            "source": "explicite" if any(x in text_norm for x in ["reel", "reelle", "physique"]) else "inférée",
                            "confiance": "high" if any(x in text_norm for x in ["reel", "reelle", "physique"]) else "medium",
                        })
        relations = []
        if interaction and xr_value in ["AR", "MR"]:
            for obj in scene_objects.get("objets_3D_principaux", [])[:5]:
                relations.append({
                    "objet": obj.get("nom", ""),
                    "objet_reel": f"{obj.get('nom', '')} réel",
                    "objet_virtuel": f"modèle 3D ou annotations XR associées à {obj.get('nom', '')}",
                    "relation": "alignement / ancrage spatial" if xr_value == "MR" else "superposition / augmentation",
                    "confiance": "medium",
                })
        return {
            "interaction_avec_reel": interaction,
            "lieu_reel": self._infer_real_place(text_norm, scene_objects),
            "indices_detection": list(dict.fromkeys(indicators)),
            "type_interaction": list(dict.fromkeys(types)) or (["interaction avec objet réel probable"] if interaction else []),
            "objets_reels": self._dedup_dicts(real_objs, "nom")[:12],
            "relations_reel_virtuel": relations,
        }

    def _infer_real_place(self, text_norm: str, scene_objects: Dict[str, Any]) -> str | None:
        for place in ["atelier", "usine", "ligne de production", "entrepot", "local technique", "zone de maintenance", "chantier", "laboratoire"]:
            if self._contains_phrase(text_norm, normalize_text(place))[0]:
                return place.title()
        if scene_objects.get("scene_principale", {}).get("nom"):
            return scene_objects["scene_principale"]["nom"]
        return None

    def _real_description(self, obj: Dict[str, Any], xr: str | None, types: List[str]) -> str:
        if xr == "AR":
            return f"Objet physique sur lequel des informations augmentées peuvent être affichées : {obj.get('description','')}"
        if xr == "MR":
            return f"Objet physique pouvant être aligné ou associé spatialement à des éléments virtuels : {obj.get('description','')}"
        return f"Objet physique identifié dans le scénario : {obj.get('description','')}"

    def _interaction_label(self, xr: str | None, types: List[str]) -> str:
        if types:
            return "; ".join(list(dict.fromkeys(types))[:3])
        if xr == "AR": return "superposition d’informations / guidage AR"
        if xr == "MR": return "alignement réel-virtuel / ancrage spatial"
        return "interaction avec objet réel"

    def analyze(self, text: str, activity_threshold: float = 0.55) -> Dict[str, Any]:
        text_norm = normalize_text(text)
        context = self.classify_context(text_norm)
        xr_val = context.get("type_XR", {}).get("valeur")
        dom_val = context.get("domaine_application", {}).get("valeur")
        activities = self.detect_activities(text_norm, threshold=activity_threshold)
        context = self._infer_context_if_missing(text_norm, context, activities)
        xr_val = context.get("type_XR", {}).get("valeur")
        dom_val = context.get("domaine_application", {}).get("valeur")
        hardware = self.detect_hardware(text_norm, xr_val, dom_val, activities)
        scene_objects = self.extract_scene_and_objects(text, text_norm, dom_val, activities)
        real_env = self.extract_real_environment(text_norm, xr_val, scene_objects)
        return {
            "texte_utilisateur": text,
            "texte_normalise": text_norm,
            "contexte_utilisation": {
                "type_XR": context.get("type_XR", {}),
                "domaine_application": context.get("domaine_application", {}),
                "activites": activities,
            },
            "environnement_immersif": hardware,
            "scene_et_objets_virtuels": scene_objects,
            "environnement_reel": real_env,
            "debug": {"scores": context.get("scores", {})},
        }

    def _dedup_dicts(self, items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for item in items:
            val = normalize_text(str(item.get(key, "")))
            if val and val not in seen:
                out.append(item); seen.add(val)
        return out


def to_pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)
