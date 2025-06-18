# 🏷️ NER Wolof - Reconnaissance d'Entités Nommées en Wolof

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Un système de reconnaissance d'entités nommées (NER) spécialement entraîné pour la langue wolof, utilisant des modèles Transformer state-of-the-art.

## 🌟 Aperçu

Ce projet propose un modèle de reconnaissance d'entités nommées pour le **wolof**, une langue parlée par plus de 12 millions de personnes principalement au Sénégal. Le modèle peut identifier et classifier automatiquement :

- 👤 **PER** - Personnes
- 🏢 **ORG** - Organisations  
- 📍 **LOC** - Lieux
- 📅 **DATE** - Dates

## 🎯 Performances

| Métrique | Validation | Test |
|----------|------------|------|
| **F1 Score** | 79.85% | 78.28% |
| **Précision** | 79.09% | 77.20% |
| **Rappel** | 80.62% | 79.38% |
| **Accuracy** | 98.11% | 97.96% |

## 🚀 Démo en Ligne

Une interface web interactive est disponible pour tester le modèle en temps réel :

```bash
streamlit run app.py
```

![Demo Screenshot](https://via.placeholder.com/800x400/4CAF50/white?text=Demo+Screenshot)

## 📦 Installation

### Prérequis

- Python 3.8+
- PyTorch
- Transformers
- Streamlit

### Installation rapide

```bash
# Cloner le repository
git clone https://github.com/votre-username/ner-wolof.git
cd ner-wolof

# Créer un environnement virtuel
python -m venv ner-env
source ner-env/bin/activate  # Linux/Mac
# ou
ner-env\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances

```txt
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
streamlit>=1.28.0
scikit-learn>=1.0.0
evaluate>=0.4.0
numpy>=1.21.0
pandas>=1.3.0
plotly>=5.0.0
requests>=2.25.0
beautifulsoup4>=4.9.0
```

## 🏃‍♂️ Utilisation Rapide

### Interface Web (Recommandé)

```bash
streamlit run app.py
```

### API Python

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Charger le modèle
model_path = "./ner-wolof-final-5epochs"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# Créer le pipeline
ner_pipeline = pipeline("ner", 
                       model=model, 
                       tokenizer=tokenizer,
                       aggregation_strategy="simple")

# Analyser un texte
text = "Aminata dem na Thiès te mu jënd ci UCAD"
results = ner_pipeline(text)

for entity in results:
    print(f"{entity['word']} → {entity['entity_group']} ({entity['score']:.3f})")
```

### Notebook Jupyter

Un notebook complet d'entraînement est disponible : [`NER WOLOF-1.ipynb`](NER%20WOLOF-1.ipynb)

## 🔧 Entraînement du Modèle

### Dataset

Le modèle est entraîné sur **MasakhaNER 2.0**, un dataset spécialisé pour les langues africaines :

- **Train** : 4,593 exemples
- **Validation** : 656 exemples  
- **Test** : 1,312 exemples

### Architecture

- **Modèle de base** : `xlm-roberta-base`
- **Optimisé** pour les langues peu dotées
- **Tokenisation** : SentencePiece (meilleure pour le wolof)

### Configuration d'entraînement

```python
training_args = TrainingArguments(
    num_train_epochs=5,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1"
)
```

## 📊 Exemples d'Utilisation

### Texte en Wolof

```python
# Exemple 1
text = "Ousmane Sonko ak Macky Sall nañu togg ci Dakar"
# Résultat: Ousmane Sonko (PER), Macky Sall (PER), Dakar (LOC)

# Exemple 2  
text = "UNESCO ak gouvernement bu Sénégal bokk ci projet bi"
# Résultat: UNESCO (ORG), Sénégal (LOC)

# Exemple 3
text = "Meeting bi dina am ci janvier 2024"
# Résultat: janvier 2024 (DATE)
```

### Texte Mixte (Wolof-Français)

Le modèle gère aussi les textes mixtes :

```python
text = "Ministre Ahmed Aidara wax na ne réunion bi dina am ci juin"
# Résultat: Ahmed Aidara (PER), juin (DATE)
```

## 🏗️ Structure du Projet

```
ner-wolof/
├── 📓 NER WOLOF-1.ipynb           # Notebook d'entraînement
├── 🎨 app.py                      # Interface Streamlit
├── 🎨 app_modern.py               # Interface moderne
├── 📁 ner-wolof-final-5epochs/    # Modèle entraîné
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   ├── label_mappings.json
│   ├── evaluation_results.json
│   ├── README.md
│   └── example_usage.py
├── 📋 requirements.txt            # Dépendances
├── 📄 README.md                   # Ce fichier
└── 📸 screenshots/                # Captures d'écran
```

## 🔬 Méthodologie

### 1. Préparation des Données

- **Dataset** : MasakhaNER 2.0 (wolof)
- **Préprocessing** : Tokenisation optimisée
- **Gestion** des mots coupés (sous-mots)

### 2. Architecture du Modèle

```python
XLM-RoBERTa Base (125M paramètres)
↓
Token Classification Head (9 classes)
↓
Labels: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-DATE, I-DATE
```

### 3. Entraînement

- **5 epochs** avec early stopping
- **Learning rate scheduling** avec warmup
- **Weight decay** pour éviter l'overfitting
- **Évaluation** à chaque epoch

### 4. Optimisations

- **Sélection automatique** du meilleur tokenizer
- **Gestion intelligente** des sous-mots
- **Post-traitement** pour fusionner les entités fragmentées

## 📈 Analyse des Résultats

### Performance par Type d'Entité

| Type | Précision | Rappel | F1-Score | Support |
|------|-----------|---------|----------|---------|
| **PER** | 80.11% | 91.89% | 85.60% | 456 |
| **LOC** | 91.22% | 79.48% | 84.95% | 536 |
| **ORG** | 79.06% | 79.28% | 79.17% | 362 |
| **DATE** | 81.16% | 66.27% | 72.96% | 169 |

### Matrice de Confusion

![Confusion Matrix](https://via.placeholder.com/400x300/2196F3/white?text=Confusion+Matrix)

## 🔄 Améliorer le Modèle

### Ajout de Données

```python
# Ajouter vos propres données d'entraînement
from datasets import Dataset

custom_data = {
    "tokens": [["Nouveau", "texte", "en", "wolof"]],
    "ner_tags": [0, 0, 0, 0]  # O, O, O, O
}

custom_dataset = Dataset.from_dict(custom_data)
```

### Fine-tuning

```python
# Continuer l'entraînement sur vos données
trainer = Trainer(
    model=model,
    train_dataset=custom_dataset,
    # ... autres paramètres
)

trainer.train()
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. **Fork** le projet
2. **Créer** une branche (`git checkout -b feature/AmazingFeature`)
3. **Commit** vos changements (`git commit -m 'Add AmazingFeature'`)
4. **Push** vers la branche (`git push origin feature/AmazingFeature`)
5. **Ouvrir** une Pull Request

### Guidelines

- ✅ Suivre le style de code existant
- ✅ Ajouter des tests pour les nouvelles fonctionnalités
- ✅ Mettre à jour la documentation
- ✅ Vérifier que tous les tests passent

## 📋 TODO

- [ ] 🌐 Déploiement sur Hugging Face Hub
- [ ] 📱 Application mobile
- [ ] 🔊 Support audio (Speech-to-NER)
- [ ] 📚 Dataset étendu avec plus d'exemples
- [ ] 🚀 Modèle plus rapide (DistilBERT)
- [ ] 🌍 Support multilingue (français-wolof)

## 🐛 Problèmes Connus

### Résolution des Erreurs Communes

```bash
# Erreur de mémoire sur CPU
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Problème de tokenisation
pip install transformers --upgrade

# Erreur Streamlit
streamlit cache clear
```

## 📚 Ressources

### Documentation

- [Transformers Documentation](https://huggingface.co/transformers/)
- [MasakhaNER Paper](https://arxiv.org/abs/2103.11811)
- [XLM-RoBERTa Paper](https://arxiv.org/abs/1911.02116)

### Datasets

- [MasakhaNER 2.0](https://huggingface.co/datasets/masakhane/masakhaner2)
- [Wolof Wikipedia](https://wo.wikipedia.org/)

### Modèles Similaires

- [Davlan/bert-base-multilingual-cased-masakhaner](https://huggingface.co/Davlan/bert-base-multilingual-cased-masakhaner)
- [microsoft/mdeberta-v3-base](https://huggingface.co/microsoft/mdeberta-v3-base)

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **MasakhaNE** pour le dataset MasakhaNER
- **Hugging Face** pour la bibliothèque Transformers
- **Communauté Wolof** pour le support linguistique
- **Université Cheikh Anta Diop** pour les ressources

## 📞 Contact

- **Auteur** : Votre Nom
- **Email** : votre.email@example.com
- **LinkedIn** : [Votre Profil](https://linkedin.com/in/votre-profil)
- **Twitter** : [@VotreHandle](https://twitter.com/VotreHandle)

## 📊 Statistiques du Projet

![GitHub stars](https://img.shields.io/github/stars/votre-username/ner-wolof)
![GitHub forks](https://img.shields.io/github/forks/votre-username/ner-wolof)
![GitHub issues](https://img.shields.io/github/issues/votre-username/ner-wolof)
![GitHub last commit](https://img.shields.io/github/last-commit/votre-username/ner-wolof)

---

<div align="center">
  <strong>🌍 Fait avec ❤️ pour la langue wolof</strong>
  <br>
  <sub>Contribuer à la préservation et à la valorisation des langues africaines</sub>
</div>
