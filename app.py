import json
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Code du numérique du Bénin", page_icon="⚖️")


@st.cache_resource
def charger_moteur():
    with open("articles.json", encoding="utf-8") as f:
        articles = json.load(f)
    textes = [a["texte"] for a in articles]
    vectoriseur = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), sublinear_tf=True, max_df=0.5)
    matrice = vectoriseur.fit_transform(textes)
    return articles, vectoriseur, matrice


articles, vectoriseur, matrice = charger_moteur()


def rechercher(question, k, seuil):
    vecteur = vectoriseur.transform([question])
    scores = cosine_similarity(vecteur, matrice)[0]
    meilleurs = scores.argsort()[::-1][:k]
    return [(articles[i], scores[i]) for i in meilleurs if scores[i] >= seuil]


st.title("Moteur de recherche du Code du numérique du Bénin")
st.caption("Loi n° 2017-20, version votée par l'Assemblée nationale. "
           "Information générale, pas un conseil juridique.")

with st.sidebar:
    st.header("Réglages")
    k = st.slider("Nombre d'articles à afficher", 1, 10, 3)
    seuil = st.slider("Seuil de pertinence", 0.0, 0.3, 0.05, 0.01)

question = st.text_input("Posez votre question, ou tapez un numéro d'article (exemple : article 45)")

if question:
    correspondance = re.fullmatch(r"\s*(?:article\s*)?(\d+)\s*", question, flags=re.IGNORECASE)

    if correspondance:
        numero = int(correspondance.group(1))
        trouves = [a for a in articles if a["numero"] == numero]
        if trouves:
            st.subheader(f"Article {numero}")
            st.write(trouves[0]["texte"])
        else:
            st.warning("Cet article n'a pas été retrouvé dans le texte utilisé.")
    else:
        resultats = rechercher(question, k, seuil)
        if not resultats:
            st.info("Aucun article du Code ne correspond à cette question.")
        else:
            st.write(f"{len(resultats)} article(s) trouvé(s) :")
            for position, (article, score) in enumerate(resultats):
                titre = f"Article {article['numero']} (pertinence : {score:.2f})"
                with st.expander(titre, expanded=(position == 0)):
                    st.write(article["texte"])