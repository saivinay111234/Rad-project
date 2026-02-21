"""
Radiology Knowledge Base for RAG (Retrieval-Augmented Generation).

Embeds key clinical guidelines (Fleischner, Lung-RADS, BI-RADS, TI-RADS, LI-RADS)
as structured text chunks and provides keyword-based retrieval to ground LLM prompts
in evidence-based medicine.

This uses a lightweight TF-IDF/keyword approach — no external vector DB required.
For production at scale, replace with a vector store (pgvector, Chroma, Weaviate).
"""

import re
import math
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guideline Text Chunks
# Each chunk is a (document_id, title, body) tuple.
# These represent simplified, structured summaries of major radiology guidelines.
# ---------------------------------------------------------------------------

_GUIDELINE_CHUNKS: List[Dict[str, str]] = [
    # ---- Fleischner Society Pulmonary Nodule Guidelines (2017) ----
    {
        "id": "fleischner_low_risk_small",
        "title": "Fleischner 2017: Low-Risk Small Solid Pulmonary Nodule",
        "body": (
            "For low-risk patients with solid pulmonary nodules <6mm: No routine follow-up recommended. "
            "Risk factors for lung cancer (smoking history, upper lobe location, irregular margins, "
            "emphysema) determine whether optional CT at 12 months is warranted. "
            "Low-risk patients have no known risk factors for lung cancer."
        ),
        "tags": ["fleischner", "pulmonary", "nodule", "lung", "low-risk", "solid", "small"],
    },
    {
        "id": "fleischner_high_risk_small",
        "title": "Fleischner 2017: High-Risk Small Solid Pulmonary Nodule",
        "body": (
            "For high-risk patients with solid pulmonary nodules <6mm: Optional CT at 12 months. "
            "High-risk patients have at least one risk factor (smoking, upper lobe location, "
            "irregular spiculated margins, emphysema, fibrosis, family history of lung cancer). "
            "Nodule size is measured as average of long and short axis diameters."
        ),
        "tags": ["fleischner", "pulmonary", "nodule", "lung", "high-risk", "solid", "small"],
    },
    {
        "id": "fleischner_medium",
        "title": "Fleischner 2017: Medium Solid Pulmonary Nodule (6-8mm)",
        "body": (
            "For solid nodules 6-8mm: CT at 6-12 months then consider CT at 18-24 months if no change. "
            "High-risk patients: CT at 6-12 months then CT at 18-24 months. "
            "For nodules >8mm: CT at 3 months, PET/CT, or tissue sampling depending on risk. "
            "Consider multidisciplinary tumor board consultation for nodules with suspicious features."
        ),
        "tags": ["fleischner", "pulmonary", "nodule", "lung", "medium", "6mm", "8mm"],
    },
    {
        "id": "fleischner_subsolid",
        "title": "Fleischner 2017: Subsolid (Ground-Glass or Part-Solid) Pulmonary Nodule",
        "body": (
            "Ground-glass nodule (GGN) <6mm: No follow-up in low-risk patients. "
            "GGN ≥6mm: CT at 6-12 months; if persistent, CT every 2 years for 5 years. "
            "Part-solid nodule <6mm: No follow-up needed. "
            "Part-solid nodule ≥6mm: CT at 3-6 months to confirm persistence, then CT every year for 5 years "
            "if stable or decreasing. If growing or solid component >6mm: PET/CT or tissue sampling."
        ),
        "tags": ["fleischner", "pulmonary", "nodule", "lung", "ground-glass", "subsolid", "GGN", "part-solid"],
    },
    # ---- Lung-RADS (ACR, 2022) ----
    {
        "id": "lung_rads_overview",
        "title": "Lung-RADS 2022: Category Overview",
        "body": (
            "Lung-RADS categories for lung cancer screening CT:\n"
            "0: Incomplete — prior CT needed for comparison or part of lung not evaluated.\n"
            "1: Negative — no nodules or definitely benign (calcified, fat-containing). Annual screening.\n"
            "2: Benign appearance — solid nodule <6mm, part-solid <6mm, GGN <30mm. Annual screening.\n"
            "3: Probably benign — solid 6-7mm, part-solid ≥6mm with solid component <6mm, GGN ≥30mm. 6-month CT.\n"
            "4A: Suspicious — solid 8-14mm, new solid <4mm. 3-month CT or PET-CT if ≥8mm.\n"
            "4B: Suspicious — solid ≥15mm, part-solid ≥8mm solid component. PET-CT or tissue sampling.\n"
            "4X: Category 3 or 4 with additional suspicious features (spiculation, upper lobe, adenopathy)."
        ),
        "tags": ["lung-rads", "lung", "screening", "LDCT", "nodule", "cancer"],
    },
    # ---- BI-RADS (ACR Breast Imaging) ----
    {
        "id": "birads_overview",
        "title": "BI-RADS 2013: Breast Imaging Reporting and Data System Categories",
        "body": (
            "BI-RADS assessment categories:\n"
            "0: Incomplete — additional imaging needed (US, magnification views).\n"
            "1: Negative — no abnormality. Annual screening.\n"
            "2: Benign — cysts, calcified fibroadenoma, lymph nodes. Annual screening.\n"
            "3: Probably benign — <2% malignancy risk. Short-term (6-month) follow-up.\n"
            "4A: Low suspicion (2-10% malignancy). Tissue sampling recommended.\n"
            "4B: Moderate suspicion (10-50% malignancy). Tissue sampling recommended.\n"
            "4C: High suspicion (50-95% malignancy). Tissue sampling recommended.\n"
            "5: Highly suggestive of malignancy (>95%). Biopsy required.\n"
            "6: Known biopsy-proven malignancy undergoing treatment."
        ),
        "tags": ["bi-rads", "birads", "breast", "mammography", "ultrasound", "MRI", "malignancy"],
    },
    # ---- TI-RADS (ACR Thyroid) ----
    {
        "id": "tirads_overview",
        "title": "ACR TI-RADS 2017: Thyroid Imaging Reporting and Data System",
        "body": (
            "ACR TI-RADS scoring:\n"
            "Composition: cystic (0), spongiform (0), mixed cystic/solid (1), solid or almost solid (2).\n"
            "Echogenicity: anechoic (0), hyperechoic/isoechoic (1), hypoechoic (2), very hypoechoic (3).\n"
            "Shape: wider-than-tall (0), taller-than-wide (3).\n"
            "Margin: smooth/ill-defined (0), lobulated/irregular (2), extrathyroidal extension (3).\n"
            "Echogenic foci: none/comet-tail (0), macrocalcifications (1), peripheral calcifications (2), "
            "punctate echogenic foci (3).\n"
            "TI-RADS 1 (0pts): benign. TI-RADS 2 (2pts): low risk. TI-RADS 3 (3pts): mildly suspicious.\n"
            "TI-RADS 4 (4-6pts): moderately suspicious. TI-RADS 5 (≥7pts): highly suspicious."
        ),
        "tags": ["ti-rads", "tirads", "thyroid", "nodule", "ultrasound", "malignancy"],
    },
    # ---- LI-RADS (ACR Liver) ----
    {
        "id": "lirads_ct_overview",
        "title": "LI-RADS v2018: Liver Imaging Reporting and Data System (CT/MRI)",
        "body": (
            "LI-RADS categories for patients at risk for hepatocellular carcinoma (HCC):\n"
            "LR-1: Definitely benign — cyst, hemangioma, focal fat deposition.\n"
            "LR-2: Probably benign — <20% HCC probability.\n"
            "LR-3: Intermediate probability — 20-50% HCC probability.\n"
            "LR-4: Probably HCC — >50% HCC probability. Consider biopsy.\n"
            "LR-5: Definitely HCC — meets imaging criteria. No biopsy required for treatment.\n"
            "LR-M: Probably or definitely malignant, not HCC specific.\n"
            "LR-TIV: Tumor in vein (portal vein thrombosis with ancillary features of HCC).\n"
            "Major features: arterial phase hyperenhancement (APHE), washout, capsule, threshold growth."
        ),
        "tags": ["li-rads", "lirads", "liver", "HCC", "hepatocellular", "carcinoma", "CT", "MRI"],
    },
    # ---- Adrenal Incidentaloma ----
    {
        "id": "adrenal_incidentaloma",
        "title": "ACR 2017: Adrenal Incidentaloma Management",
        "body": (
            "Adrenal incidentaloma evaluation:\n"
            "<10 HU unenhanced CT: lipid-rich adenoma (benign). No follow-up needed.\n"
            "10-20 HU: indeterminate. Consider adrenal-protocol CT or MRI with chemical shift.\n"
            ">20 HU or >4cm: higher concern for malignancy or pheochromocytoma. Biochemical evaluation. "
            "Surgical consultation.\n"
            "Absolute contrast washout >60% or relative washout >40%: consistent with adenoma.\n"
            "All adrenal incidentalomas: biochemical evaluation for pheochromocytoma, hyperaldosteronism "
            "(if hypertensive), and Cushing syndrome."
        ),
        "tags": ["adrenal", "incidentaloma", "adenoma", "pheochromocytoma", "CT", "washout"],
    },
    # ---- Pulmonary Embolism ----
    {
        "id": "pe_wells_criteria",
        "title": "Wells Criteria for Pulmonary Embolism Pre-test Probability",
        "body": (
            "Wells PE score criteria:\n"
            "Clinical signs/symptoms of DVT: 3 points\n"
            "Alternative diagnosis less likely than PE: 3 points\n"
            "Heart rate >100 bpm: 1.5 points\n"
            "Immobilization ≥3 days or surgery in past 4 weeks: 1.5 points\n"
            "Prior DVT/PE: 1.5 points\n"
            "Hemoptysis: 1 point\n"
            "Malignancy (treatment within 6 months or palliative): 1 point\n"
            "Score ≤4: PE unlikely — D-dimer test appropriate\n"
            "Score >4: PE likely — proceed to CT pulmonary angiography (CTPA)"
        ),
        "tags": ["pulmonary", "embolism", "PE", "DVT", "Wells", "CTPA", "D-dimer"],
    },
]


# ---------------------------------------------------------------------------
# TF-IDF Style Retrieval
# ---------------------------------------------------------------------------

class RadiologyKnowledgeBase:
    """
    Lightweight keyword-based knowledge retrieval for radiology guidelines.

    Uses term frequency (TF) weighted by inverse document frequency (IDF)
    to rank guideline chunks by relevance to a query.
    """

    def __init__(self, chunks: Optional[List[Dict[str, str]]] = None):
        self._chunks = chunks or _GUIDELINE_CHUNKS
        self._tokenized: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._build_index()
        logger.info("RadiologyKnowledgeBase built with %d chunks", len(self._chunks))

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase and split text into tokens."""
        return re.findall(r"[a-zA-Z0-9\-]+", text.lower())

    def _build_index(self) -> None:
        """Pre-compute tokenized docs and IDF scores."""
        n_docs = len(self._chunks)
        # Tokenize: title + body + tags
        for chunk in self._chunks:
            text = f"{chunk['title']} {chunk['body']} {' '.join(chunk.get('tags', []))}"
            self._tokenized.append(self._tokenize(text))

        # Compute IDF
        all_terms: Dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                all_terms[term] = all_terms.get(term, 0) + 1

        for term, doc_freq in all_terms.items():
            self._idf[term] = math.log(n_docs / (1 + doc_freq))

    def _score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """Compute TF-IDF dot product between query and document."""
        doc_freq = Counter(doc_tokens)
        doc_len = len(doc_tokens) or 1
        score = 0.0
        for qt in query_tokens:
            tf = doc_freq.get(qt, 0) / doc_len
            idf = self._idf.get(qt, 0.0)
            score += tf * idf
        return score

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve the most relevant guideline text chunks for a query.

        Args:
            query: Natural language query (e.g., "6mm pulmonary nodule low risk").
            top_k: Number of top chunks to return.

        Returns:
            List of formatted guideline text strings, ready to prepend to LLM prompts.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[float, int]] = []
        for idx, doc_tokens in enumerate(self._tokenized):
            s = self._score(query_tokens, doc_tokens)
            if s > 0:
                scores.append((s, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_indices = [idx for _, idx in scores[:top_k]]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                f"[Guideline: {chunk['title']}]\n{chunk['body']}"
            )

        logger.debug("Knowledge retrieval for query %r: %d results", query[:50], len(results))
        return results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_knowledge_base: Optional[RadiologyKnowledgeBase] = None


def get_knowledge_base() -> RadiologyKnowledgeBase:
    """Return a module-level singleton RadiologyKnowledgeBase."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = RadiologyKnowledgeBase()
    return _knowledge_base
