import json
import time
import re
import streamlit as st
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Assistant Code du numérique du Bénin")

MODELES = ["gemini-3.8-flash"]
SEUIL = 0.05
MAX_QUESTIONS = 20

CONSIGNE = """Tu es un assistant qui informe sur le Code du numérique du Bénin (loi n° 2017-20).
Règles :
- Réponds UNIQUEMENT à partir des extraits d'articles fournis ci-dessous.
- Cite le numéro de chaque article utilisé, sous la forme (Article X).
- Si les extraits ne permettent pas de répondre, dis-le clairement et ne devine pas.
- Réponds en français, de façon simple et courte.
- Tu informes : tu ne donnes pas de conseil juridique personnalisé.

CONVERSATION JUSQU'ICI :
{historique}

EXTRAITS :
{extraits}

QUESTION : {question}
"""


@st.cache_resource
def charger_moteur():
    with open("articles.json", encoding="utf-8") as f:
        articles = json.load(f)
    textes = [a["texte"] for a in articles]
    vectoriseur = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), sublinear_tf=True, max_df=0.5)
    matrice = vectoriseur.fit_transform(textes)
    return articles, vectoriseur, matrice


@st.cache_resource
def charger_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


articles, vectoriseur, matrice = charger_moteur()
client = charger_client()


def rechercher(requete, k=4):
    vecteur = vectoriseur.transform([requete])
    scores = cosine_similarity(vecteur, matrice)[0]
    meilleurs = scores.argsort()[::-1][:k]
    return [articles[i] for i in meilleurs if scores[i] >= SEUIL]


def demander_au_modele(consigne, essais=4):
    st.session_state["derniere_erreur"] = ""
    for modele in MODELES:
        for essai in range(essais):
            try:
                reponse = client.models.generate_content(model=modele, contents=consigne)
                if reponse.text:
                    return reponse.text
                st.session_state["derniere_erreur"] = "Le modèle a renvoyé une réponse vide."
                return None
            except Exception as e:
                st.session_state["derniere_erreur"] = str(e)[:300]
                print("ERREUR GEMINI :", str(e)[:500])
                if "503" in str(e) or "429" in str(e):
                    time.sleep(5 * (2 ** essai))
                else:
                    raise
    return None

def repondre(question, messages):
    questions_precedentes = [m["content"] for m in messages if m["role"] == "user"]
    requete = question
    if len(question.split()) < 6 and questions_precedentes:
        requete = questions_precedentes[-1] + " " + question

    trouves = rechercher(requete)
    if not trouves:
        return "Je n'ai trouvé aucun article du Code qui corresponde à cette question.", []

    historique = "\n".join(
        ("Utilisateur : " if m["role"] == "user" else "Assistant : ") + m["content"]
        for m in messages[-6:]
    ) or "(début de la conversation)"
    extraits = "\n\n".join(a["texte"][:1500] for a in trouves)

    texte = demander_au_modele(
        CONSIGNE.format(historique=historique, extraits=extraits, question=question))
    if texte is None:
        texte = ("L'assistant IA est momentanément indisponible. "
                 "Voici les articles les plus pertinents :")
    return texte, trouves

def mettre_en_forme(texte):
    texte = re.sub(r"\s*([▪•])\s*", r"\n\n\1 ", texte)
    texte = re.sub(r"\s(\d{1,2})-\s", r"\n\n\1- ", texte)
    texte = re.sub(r"\s-\s(?=[A-ZÀ-Ý])", "\n\n- ", texte)
    return texte
    
def afficher(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Voir les articles consultés"):
                for a in message["sources"]:
                    st.markdown(f"**Article {a['numero']}**")
                    st.markdown(mettre_en_forme(a["texte"]))


with st.sidebar:
    st.header("À propos")
    st.write("Assistant d'information sur le Code du numérique du Bénin "
             "(loi n° 2017-20, version votée par l'Assemblée nationale). "
             "Il répond à partir des articles du Code. "
             "Information générale, pas un conseil juridique.")
    if st.button("Nouvelle conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("Assistant du Code du numérique du Bénin")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    afficher(m)

question = st.chat_input("Posez votre question sur le Code du numérique...")

if question:
    nb_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
    if nb_questions >= MAX_QUESTIONS:
        st.warning("Limite de questions atteinte pour cette conversation. "
                   "Cliquez sur « Nouvelle conversation » pour continuer.")
    else:
        precedents = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        afficher(st.session_state.messages[-1])
        with st.spinner("Recherche dans le Code..."):
            reponse, sources = repondre(question, precedents)
        reponse_msg = {"role": "assistant", "content": reponse, "sources": sources}
        st.session_state.messages.append(reponse_msg)
        afficher(reponse_msg)
if st.session_state.get("derniere_erreur"):
    st.sidebar.caption("Détail technique (temporaire) : " + st.session_state["derniere_erreur"])
