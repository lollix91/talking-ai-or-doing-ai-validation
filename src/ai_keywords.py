"""Dizionario di keyword per misurare l'adozione di IA nei 10-K.

Basato su:
  - AIDI framework (American Impact Review, 2025)
  - "Talking AI, Seeing Returns" (Indiana University, 2025)
  - Baruffaldi et al. (OECD, 2020) per la riduzione dei falsi positivi

Organizzato in tier per specificita'. I termini ambigui (alto rischio di
falsi positivi) richiedono validazione del contesto (vedi nlp.py).
"""
from __future__ import annotations

# Tier 1: termini core AI (alta specificita', nessun filtro di contesto)
TIER1_CORE = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "neural networks",
    "large language model",
    "large language models",
    "natural language processing",
    "computer vision",
    "generative ai",
    "reinforcement learning",
    "supervised learning",
    "unsupervised learning",
    "convolutional neural network",
    "recurrent neural network",
    "foundation model",
    "foundation models",
]

# Tier 2: tecniche e applicazioni ML (specificita' medio-alta)
TIER2_TECHNIQUES = [
    "random forest",
    "decision tree",
    "decision trees",
    "gradient boosting",
    "pattern recognition",
    "predictive analytics",
    "predictive maintenance",
    "image recognition",
    "speech recognition",
    "sentiment analysis",
    "chatbot",
    "chatbots",
    "facial recognition",
    "optical character recognition",
    "anomaly detection",
    "recommendation engine",
    "recommendation system",
    "recommender system",
]

# Tier 3: termini emergenti / generativi (post-2020)
TIER3_EMERGING = [
    "chatgpt",
    "gpt-3",
    "gpt-4",
    "gpt",
    "generative model",
    "generative models",
    "text generation",
    "image generation",
    "fine-tuning",
    "fine tuning",
    "hallucination",
    "prompt engineering",
    "large language",
    "llm",
]

# Termini ambigui: contati solo se compaiono vicino a un termine core
# (finestra di contesto, approccio "Talking AI, Seeing Returns").
AMBIGUOUS = [
    "algorithm",
    "algorithms",
    "automation",
    "robotics",
    "autonomous",
    "cognitive computing",
    "robotic process automation",
    "rpa",
]

# Parole di contesto AI per la validazione dei termini ambigui.
# Devono essere specifiche del dominio AI: parole generiche (data, model,
# learn, predict) renderebbero il controllo di contesto quasi inutile
# perche' compaiono ovunque in un 10-K.
CONTEXT_WORDS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "neural networks",
    "ai",
    "ml",
    "intelligent",
    "cognitive",
    "gpt",
    "llm",
]

# Acronimi che richiedono confini di parola (evita match in sottostringhe).
WORD_BOUNDARY_ACRONYMS = ["ai", "ml", "llm", "gpt", "rpa", "nlp", "cnn", "rnn"]

ALL_KEYWORDS = TIER1_CORE + TIER2_TECHNIQUES + TIER3_EMERGING

# Acronimi contati CASE-SENSITIVE (solo maiuscolo/forma canonica), perche'
# dal 2018 circa i 10-K usano prevalentemente "AI" invece della forma estesa.
# "AI" subito dopo "artificial intelligence (" non viene contato (e' la
# definizione dell'acronimo, gia' contata come forma estesa).
ACRONYMS = ["AI", "GenAI"]

# Termini entrati nel lessico corrente solo di recente (generative/LLM):
# esclusi dal dizionario "fisso" usato per verificare che l'andamento
# temporale dello score non sia un artefatto dell'evoluzione del lessico.
RECENT_TERMS = [
    "large language model",
    "large language models",
    "generative ai",
    "foundation model",
    "foundation models",
] + TIER3_EMERGING + ["GenAI"]
FIXED_KEYWORDS = [t for t in ALL_KEYWORDS + ACRONYMS if t not in RECENT_TERMS]

TIER_OF = {
    **{t: "tier1" for t in TIER1_CORE},
    **{t: "acronym" for t in ACRONYMS},
    **{t: "tier2" for t in TIER2_TECHNIQUES},
    **{t: "tier3" for t in TIER3_EMERGING},
    **{t: "ambiguous" for t in AMBIGUOUS},
}
