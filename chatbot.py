"""
Chatbot RAG - Code du numérique de la République du Bénin
Pipeline : PDF -> découpage par article -> index TF-IDF -> recherche top-k -> LLM (Claude)
Usage :
    pip install pypdf scikit-learn anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python chatbot.py code_numerique_benin.pdf
"""
import re
import sys
import unicodedata
from dataclasses import dataclass

import anthropic
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "claude-sonnet-5-5"
TOP_K = 5
SEUIL_PERTINENCE = 0.05  # en dessous : on considère qu'aucun article ne répond


# ---------------------------------------------------------------- 1. INGESTION
@dataclass
class Chunk:
    article: str      # ex. "Article 12"
    contexte: str     # ex. "LIVRE III > TITRE I"
    texte: str


RE_ARTICLE = re.compile(r"^Article\s+(premier|1er|\d+)\b", re.IGNORECASE)
RE_STRUCTURE = re.compile(r"^(LIVRE|TITRE|CHAPITRE|SECTION)\b", re.IGNORECASE)


def extraire_texte(chemin_pdf: str) -> str:
    lecteur = PdfReader(chemin_pdf)
    return "\n".join(page.extract_text() or "" for page in lecteur.pages)


def decouper_par_article(texte: str) -> list[Chunk]:
    """Un chunk = un article de loi, avec son contexte (livre/titre/chapitre).
    Découper par unité juridique naturelle donne de bien meilleurs résultats
    qu'un découpage en blocs de taille fixe."""
    chunks, structure = [], {}
    article, buffer = None, []

    def clore():
        if article and buffer:
            ctx = " > ".join(structure[k] for k in sorted(structure))
            chunks.append(Chunk(article, ctx, " ".join(buffer).strip()))

    for ligne in texte.splitlines():
        ligne = ligne.strip()
        if not ligne:
            continue
        m_art = RE_ARTICLE.match(ligne)
        m_str = RE_STRUCTURE.match(ligne)
        if m_art:
            clore()
            article, buffer = f"Article {m_art.group(1)}", [ligne]
        elif m_str:
            niveau = ["LIVRE", "TITRE", "CHAPITRE", "SECTION"].index(m_str.group(1).upper())
            structure = {k: v for k, v in structure.items() if k < niveau}
            structure[niveau] = ligne[:80]
            if article:
                buffer.append(ligne)
        elif article:
            buffer.append(ligne)
    clore()
    return chunks


# ----------------------------------------------------------------- 2. INDEXATION
def normaliser(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


class Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectoriseur = TfidfVectorizer(
            preprocessor=normaliser, ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrice = self.vectoriseur.fit_transform(
            [f"{c.contexte} {c.texte}" for c in chunks]
        )

    # ------------------------------------------------------------ 3. RECHERCHE
    def rechercher(self, question: str, k: int = TOP_K):
        q = self.vectoriseur.transform([question])
        scores = cosine_similarity(q, self.matrice).ravel()
        meilleurs = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in meilleurs if scores[i] >= SEUIL_PERTINENCE]


# ---------------------------------------------------------------- 4. GENERATION
SYSTEM = """Tu es un assistant d'information sur le Code du numérique de la République du Bénin.
Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits fournis dans <extraits>. N'invente rien.
2. Cite systématiquement les articles utilisés, ex. (Article 12).
3. Si les extraits ne permettent pas de répondre, dis-le clairement et invite à consulter le texte officiel.
4. Réponds en français, de façon claire et accessible à un non-juriste.
5. Termine par un rappel bref : ces informations ne constituent pas un avis juridique."""


def construire_prompt(question: str, resultats) -> str:
    extraits = "\n\n".join(
        f"[{c.article} | {c.contexte}]\n{c.texte}" for c, _ in resultats
    )
    return f"<extraits>\n{extraits}\n</extraits>\n\nQuestion : {question}"


def repondre(client, index: Index, question: str, historique: list[dict]) -> str:
    resultats = index.rechercher(question)
    if not resultats:
        return ("Je n'ai trouvé aucun article du Code du numérique correspondant à votre question. "
                "Pouvez-vous la reformuler ?")
    messages = historique[-6:] + [{"role": "user", "content": construire_prompt(question, resultats)}]
    rep = client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM, messages=messages)
    texte = rep.content[0].text
    # On mémorise la question simple (sans les extraits) pour garder l'historique léger
    historique += [{"role": "user", "content": question}, {"role": "assistant", "content": texte}]
    return texte


# --------------------------------------------------------------------- MAIN
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage : python chatbot.py code_numerique_benin.pdf")
    chunks = decouper_par_article(extraire_texte(sys.argv[1]))
    print(f"{len(chunks)} articles indexés.")
    index, client, historique = Index(chunks), anthropic.Anthropic(), []
    print("Posez vos questions sur le Code du numérique (Ctrl+C pour quitter).\n")
    while True:
        try:
            q = input("Vous > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q:
            print("\nBot >", repondre(client, index, q, historique), "\n")


if __name__ == "__main__":
    main()
