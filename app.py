import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import requests
from bs4 import BeautifulSoup
import json
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="NER Wolof - Analyse Intelligente",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
/* Variables CSS pour la cohérence des couleurs */
:root {
    --primary-color: #1f77b4;
    --secondary-color: #ff7f0e;
    --success-color: #2ca02c;
    --danger-color: #d62728;
    --warning-color: #ff7f0e;
    --info-color: #17a2b8;
    --light-bg: #f8f9fa;
    --dark-bg: #343a40;
    --border-color: #e9ecef;
}

/* Style général de l'application */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

.main-header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
}

.main-header p {
    font-size: 1.2rem;
    opacity: 0.9;
    margin: 0;
}

/* Styles pour les entités avec meilleur contraste */
.entity-PER {
    background: linear-gradient(135deg, #ffebee, #f8bbd9);
    color: #880e4f;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 600;
    margin: 2px;
    display: inline-block;
    border: 2px solid #ad1457;
    box-shadow: 0 2px 4px rgba(136, 14, 79, 0.1);
}

.entity-LOC {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    color: #1b5e20;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 600;
    margin: 2px;
    display: inline-block;
    border: 2px solid #388e3c;
    box-shadow: 0 2px 4px rgba(27, 94, 32, 0.1);
}

.entity-ORG {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    color: #0d47a1;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 600;
    margin: 2px;
    display: inline-block;
    border: 2px solid #1976d2;
    box-shadow: 0 2px 4px rgba(13, 71, 161, 0.1);
}

.entity-DATE {
    background: linear-gradient(135deg, #fff3e0, #ffcc02);
    color: #e65100;
    padding: 4px 8px;
    border-radius: 8px;
    font-weight: 600;
    margin: 2px;
    display: inline-block;
    border: 2px solid #f57c00;
    box-shadow: 0 2px 4px rgba(230, 81, 0, 0.1);
}

/* Zone de résultat améliorée */
.result-text {
    font-size: 1.2rem;
    line-height: 1.8;
    padding: 1.5rem;
    border: none;
    border-radius: 12px;
    background: #1e1e2f;  /* fond sombre */
    color: #f8f8f8;        /* texte clair */
    min-height: 120px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    border-left: 4px solid var(--primary-color);
    overflow-x: auto;
}


/* Légende moderne */
.legend {
    padding: 1.5rem;
    background: linear-gradient(145deg, #f8f9fa, #e9ecef);
    border-radius: 12px;
    margin: 1rem 0;
    border: 1px solid var(--border-color);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.legend h4 {
    color: #495057;
    margin-bottom: 1rem;
    font-weight: 600;
}

/* Cards pour les statistiques */
.stat-card {
    background: grey;
    padding: 1.5rem;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    border-left: 4px solid var(--primary-color);
    margin: 0.5rem 0;
}

.stat-card h3 {
    color: var(--primary-color);
    font-size: 2rem;
    margin: 0;
    font-weight: 700;
}

.stat-card p {
    color: #6c757d;
    margin: 0.5rem 0 0 0;
    font-weight: 500;
}

/* Sidebar améliorée */
.sidebar-content {
    padding: 1rem;
}

/* Boutons stylisés */
.stButton > button {
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

/* Animation de chargement */
.loading-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .main-header p {
        font-size: 1rem;
    }
    
    .result-text {
        font-size: 1rem;
        padding: 1rem;
    }
}

/* Animation pour les entités */
.entity-PER, .entity-LOC, .entity-ORG, .entity-DATE {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Style pour les métriques */
.metric-container {
    background: #f1f1f1;
    color: #222;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    margin: 0.25rem;
    text-align: center;
}

/* Table style amélioré */
.entity-table {
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

.entity-row {
    padding: 0.75rem;
    border-bottom: 1px solid #e9ecef;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.entity-row:hover {
    background-color: #f8f9fa;
}

.entity-row:last-child {
    border-bottom: none;
}

.entity-PER, .entity-LOC, .entity-ORG, .entity-DATE {
    font-size: 1.05rem;
    line-height: 1.6;
    font-family: 'Segoe UI', sans-serif;
    color: #111 !important; /* plus foncé */
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ner_model():
    """Charger le modèle NER entraîné"""
    model_path = "/Users/macbook/Desktop/ner-wolof-final-5epochs"
    
    try:
        device = torch.device("cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        model = model.to(device)
        
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1
        )
        
        return ner_pipeline
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.error("Vérifiez que le modèle est bien sauvegardé dans le répertoire spécifié")
        return None

def highlight_entities(text, entities):
    """Colorer les entités dans le texte avec animation"""
    if not entities:
        return text
    
    entities_sorted = sorted(entities, key=lambda x: x['start'], reverse=True)
    
    result_text = text
    for entity in entities_sorted:
        start = entity['start']
        end = entity['end']
        word = entity['word']
        entity_type = entity['entity_group']
        
        colored_word = f'<span class="entity-{entity_type}" title="{entity_type} - Confiance: {entity["score"]:.2%}">{word}</span>'
        result_text = result_text[:start] + colored_word + result_text[end:]
    
    return result_text

def create_entity_chart(entities):
    """Créer un graphique des entités détectées"""
    if not entities:
        return None
    
    # Compter les entités par type
    entity_counts = {}
    for entity in entities:
        entity_type = entity['entity_group']
        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    # Créer le graphique en barres
    fig = px.bar(
        x=list(entity_counts.keys()),
        y=list(entity_counts.values()),
        title="Distribution des Types d'Entités",
        labels={'x': 'Type d\'Entité', 'y': 'Nombre'},
        color=list(entity_counts.keys()),
        color_discrete_map={
            'PER': '#ad1457',
            'LOC': '#388e3c', 
            'ORG': '#1976d2',
            'DATE': '#f57c00'
        }
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_confidence_chart(entities):
    """Créer un graphique de confiance des entités"""
    if not entities:
        return None
    
    df = pd.DataFrame([{
        'Entité': entity['word'],
        'Type': entity['entity_group'],
        'Confiance': entity['score']
    } for entity in entities])
    
    fig = px.scatter(
        df, 
        x='Entité', 
        y='Confiance',
        color='Type',
        size='Confiance',
        title="Niveau de Confiance par Entité",
        color_discrete_map={
            'PER': '#ad1457',
            'LOC': '#388e3c', 
            'ORG': '#1976d2',
            'DATE': '#f57c00'
        }
    )
    
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def scrape_presidency_texts():
    """Récupérer des textes en wolof depuis le site de la Présidence"""
    try:
        url = "https://www.presidence.sn/wo/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        texts = []
        for element in soup.find_all(['p', 'div', 'article']):
            text = element.get_text(strip=True)
            if len(text) > 30 and any(char.isalpha() for char in text):
                text = re.sub(r'\s+', ' ', text).strip()
                texts.append(text)
        
        return texts[:5]
        
    except Exception as e:
        st.warning(f"Impossible de récupérer les textes: {e}")
        return []

def main():
    # En-tête principal avec design moderne
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ NER Wolof - Analyse Intelligente</h1>
        <p>Reconnaissance d'entités nommées avancée pour la langue wolof</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations et contrôles
    with st.sidebar:
        st.markdown("### 🎛️ Panneau de Contrôle")
        
        # Informations sur le modèle
        with st.expander("ℹ️ Informations sur le Modèle", expanded=False):
            st.markdown("""
            **Modèle:** XLM-RoBERTa fine-tuné  
            **Dataset:** MasakhaNER  
            **Langues:** Wolof  
            **Types d'entités:** PER, LOC, ORG, DATE  
            **Précision:** ~85%
            """)
        
        # Statistiques de session
        if 'analysis_count' not in st.session_state:
            st.session_state.analysis_count = 0
        
        st.markdown("### 📊 Statistiques de Session")
        st.metric("Analyses effectuées", st.session_state.analysis_count)
        
        # Paramètres avancés
        st.markdown("### ⚙️ Paramètres")
        confidence_threshold = st.slider(
            "Seuil de confiance minimum",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Entités avec une confiance inférieure seront filtrées"
        )
        
        show_charts = st.checkbox("Afficher les graphiques", value=True)
        show_detailed_results = st.checkbox("Résultats détaillés", value=True)
        
        # Bouton de réinitialisation
        if st.button("🔄 Réinitialiser Session"):
            st.session_state.analysis_count = 0
            st.session_state.input_text = ""
            st.rerun()
    
    # Charger le modèle
    ner_pipeline = load_ner_model()
    
    if ner_pipeline is None:
        st.stop()
    
    st.success("✅ Modèle NER chargé avec succès!")
    
    # Légende moderne des couleurs
    st.markdown("""
    <div class="legend">
        <h4>🎨 Légende des Types d'Entités</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
            <span class="entity-PER">👤 PER - Personnes</span>
            <span class="entity-LOC">📍 LOC - Lieux</span>
            <span class="entity-ORG">🏢 ORG - Organisations</span>
            <span class="entity-DATE">📅 DATE - Dates</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Zone principale d'analyse
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Zone de Saisie")
        
        # Exemples rapides avec un design amélioré
        st.markdown("**🚀 Exemples rapides:**")
        examples_col1, examples_col2, examples_col3 = st.columns(3)
        
        examples = [
            "Ousmane Sonko ak Macky Sall nañu topp ci Dakar.",
            "Aminata dem na Thiès ci septembre 2024.",
            "UNESCO ak UCAD nañu bokk ci projet bi."
        ]
        
        for i, (col, example) in enumerate(zip([examples_col1, examples_col2, examples_col3], examples)):
            if col.button(f"📝 Exemple {i+1}", use_container_width=True, key=f"example_{i}"):
                st.session_state.input_text = example
        
        # Zone de texte principale
        input_text = st.text_area(
            "**Tapez votre texte en wolof ici:**",
            value=st.session_state.get('input_text', ''),
            height=150,
            placeholder="Exemple: Président bi dem na Dakar ci 15 janvier 2024...",
            key="text_input"
        )
        
        # Boutons d'action
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            analyze_button = st.button("🔍 Analyser le Texte", type="primary", use_container_width=True)
        
        with action_col2:
            if st.button("🌐 Récupérer depuis presidence.sn", use_container_width=True):
                with st.spinner("Récupération en cours..."):
                    texts = scrape_presidency_texts()
                    if texts:
                        scraped_text = texts[0][:500]
                        st.session_state.input_text = scraped_text
                        st.rerun()
                    else:
                        st.warning("Aucun texte récupéré.")
        
        with action_col3:
            if st.button("🗑️ Effacer", use_container_width=True):
                st.session_state.input_text = ""
                st.rerun()
    
    with col2:
        st.markdown("### 📈 Aperçu Rapide")
        if input_text.strip():
            word_count = len(input_text.split())
            char_count = len(input_text)
            
            st.metric("Mots", word_count)
            st.metric("Caractères", char_count)
            
            # Estimation du temps de traitement
            estimated_time = max(0.5, word_count * 0.1)
            st.metric("Temps estimé", f"{estimated_time:.1f}s")
    
    # Analyse du texte
    if (analyze_button or input_text != st.session_state.get('last_analyzed_text', '')) and input_text.strip():
        st.session_state.last_analyzed_text = input_text
        st.session_state.analysis_count += 1
        
        st.markdown("---")
        st.markdown("## 🎯 Résultats de l'Analyse")
        
        with st.spinner("🔄 Analyse en cours..."):
            try:
                # Simulation d'un temps de traitement réaliste
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Analyse NER
                entities = ner_pipeline(input_text)
                
                # Filtrer par seuil de confiance
                filtered_entities = [e for e in entities if e['score'] >= confidence_threshold]
                
                progress_bar.empty()
                
                if filtered_entities:
                    # Texte avec entités colorées
                    highlighted_text = highlight_entities(input_text, filtered_entities)
                    st.markdown("### 📝 Texte Analysé")
                    st.markdown(
                        f'<div class="result-text">{highlighted_text}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Métriques rapides
                    st.markdown("### 📊 Résumé des Entités")
                    entity_counts = {}
                    for entity in filtered_entities:
                        entity_type = entity['entity_group']
                        entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    # Affichage des métriques en colonnes
                    metrics_cols = st.columns(len(entity_counts) if entity_counts else 1)
                    for i, (entity_type, count) in enumerate(entity_counts.items()):
                        with metrics_cols[i]:
                            st.markdown(f"""
                            <div class="metric-container">
                                <h3>{count}</h3>
                                <p>{entity_type}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Graphiques (si activés)
                    if show_charts:
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            entity_chart = create_entity_chart(filtered_entities)
                            if entity_chart:
                                st.plotly_chart(entity_chart, use_container_width=True)
                        
                        with chart_col2:
                            confidence_chart = create_confidence_chart(filtered_entities)
                            if confidence_chart:
                                st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    # Résultats détaillés (si activés)
                    if show_detailed_results:
                        st.markdown("### 📋 Détail des Entités")
                        
                        # Créer un DataFrame pour un affichage tabulaire
                        df_entities = pd.DataFrame([{
                            'N°': i + 1,
                            'Entité': entity['word'],
                            'Type': entity['entity_group'],
                            'Confiance': f"{entity['score']:.2%}",
                            'Position': f"{entity['start']}-{entity['end']}"
                        } for i, entity in enumerate(filtered_entities)])
                        
                        st.dataframe(
                            df_entities,
                            use_container_width=True,
                            hide_index=True
                        )
                
                else:
                    st.markdown(
                        f'<div class="result-text">{input_text}</div>',
                        unsafe_allow_html=True
                    )
                    st.info("ℹ️ Aucune entité nommée détectée avec le seuil de confiance défini.")
                    
                    if entities:  # Si des entités existent mais sont filtrées
                        st.warning(f"💡 {len(entities)} entité(s) détectée(s) mais filtrée(s) par le seuil de confiance ({confidence_threshold:.1%})")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {e}")
    
    elif input_text.strip() == "":
        st.info("👆 Saisissez un texte en wolof ci-dessus pour commencer l'analyse.")
    
    # Pied de page avec informations supplémentaires
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("""
        **🔬 Technologie**
        - Transformers
        - PyTorch
        - XLM-RoBERTa
        """)
    
    with footer_col2:
        st.markdown("""
        **📊 Données**
        - Dataset MasakhaNER
        - Langue Wolof
        - 4 types d'entités
        """)
    
    with footer_col3:
        st.markdown("""
        **🌟 Fonctionnalités**
        - Analyse en temps réel
        - Visualisations interactives
        - Interface responsive
        """)

if __name__ == "__main__":
    main()
