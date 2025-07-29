# A Novel Emotion-Aware Movie Recommendation System Using Hybrid Emotional States

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)

## 🎬 Overview

Traditional movie recommendation systems rely on static user preferences or collaborative filtering methods, which fail to capture the dynamic and nuanced nature of human emotions. This project introduces a novel recommendation system that utilizes **hybrid emotions**—mixtures of basic affective states—to better reflect real human moods and provide more personalized movie recommendations.

Unlike systems that derive mood from biometric signals, this system allows users to manually input their emotional state through an intuitive interface, which is then processed using advanced Natural Language Processing (NLP) techniques to generate emotionally congruent movie recommendations.

## 🌟 Key Features

- **Hybrid Emotion Recognition**: Captures complex emotional states like "nostalgic happiness," "anxious enthusiasm," or "peaceful sorrow"
- **Manual Mood Input**: User-friendly interface for expressing emotions in natural language
- **Advanced NLP Processing**: Utilizes Sentence-BERT (SBERT) embeddings for semantic understanding
- **Two-Stage Classification**: Basic mood assignment followed by complex mood refinement
- **Personalized Recommendations**: Emotionally aware movie suggestions based on current mood
- **Comprehensive Evaluation**: F1-score based performance metrics with macro-average assessment

## 🏗️ System Architecture

The system follows an 8-stage methodology:

1. **Data Preprocessing**: Text cleaning and standardization
2. **NLP Value Enrichment**: Sentiment analysis and emotion extraction using BERT-based models
3. **Visualization**: Pattern discovery through data visualization
4. **Exploratory Data Analysis (EDA)**: Statistical analysis and anomaly detection
5. **Feature Engineering & PCA**: Dimensionality reduction and feature optimization
6. **Basic Mood Labelling**: Assignment of fundamental emotional categories
7. **Complex Mood Labelling**: Refined emotional state classification
8. **Evaluation**: Performance assessment using appropriate metrics

## 🛠️ Technical Implementation

### Core Technologies

- **Python 3.8+**
- **Sentence-BERT (all-MiniLM-L6-v2)**: For semantic embeddings
- **TextBlob**: Sentiment analysis
- **spaCy (en_core_web_sm)**: Named Entity Recognition
- **scikit-learn**: Machine learning utilities
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization

### Model Architecture

The system employs a hybrid approach combining:

1. **Keyword-based Heuristics**: For initial mood classification
2. **Semantic Similarity Calculation**: Using SBERT embeddings and cosine similarity
3. **Sentiment Adjustment**: Polarity-based mood score refinement

### Mood Categories

**Basic Moods (10 categories):**
- Happy, Sad, Angry, Fearful, Excited, Calm, Bored, Surprised, Disgusted, Neutral

**Complex Moods (24 refined states):**
- Melancholy, Euphoric, Nostalgic, Anxious, Hopeful, Serene, Contemplative, and more

## 📊 Dataset

The dataset undergoes three transformation stages:

1. **Original Dataset**: Movie titles and overviews with basic mood labels
2. **Preprocessed Dataset**: Cleaned text with SBERT embeddings
3. **Final Enriched Dataset**: Complex mood assignments and refined classifications

### Data Processing Pipeline

```
Raw Movie Data → Text Cleaning → Sentiment Analysis → NER → SBERT Embedding → Mood Classification → Recommendation Generation
```

## 🎯 Performance Results

The model achieved the following F1-scores for basic mood prediction:

| Mood Category | F1-Score |
|---------------|----------|
| Happy         | 0.6435   |
| Sad           | 0.9216   |
| Angry         | 0.7563   |
| Fearful       | 0.5213   |
| **Macro Avg** | **0.7111** |

### Comparison with Baseline Models

| Model | Happy | Sad | Angry | Fear | Macro Avg |
|-------|-------|-----|-------|------|-----------|
| SVM + TF-IDF | 0.70 | 0.66 | 0.75 | 0.72 | 0.67 |
| Random Forest + BoW | 0.70 | 0.65 | 0.75 | 0.72 | 0.68 |
| Random Forest + TF-IDF | 0.69 | 0.67 | 0.72 | 0.70 | 0.68 |
| **Our Model** | **0.64** | **0.92** | **0.75** | **0.52** | **0.71** |

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install sentence-transformers textblob spacy
pip install jupyter notebook
python -m spacy download en_core_web_sm
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-aware-movie-recommendation.git
cd emotion-aware-movie-recommendation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook emotion_movie_recommendation.ipynb
```

### Usage

1. **Load the dataset**: The system uses `dataset.csv` containing movie information
2. **Input your mood**: Enter your current emotional state in natural language
3. **Get recommendations**: Receive personalized movie suggestions based on your mood
4. **Explore results**: Analyze the emotion-movie mappings and recommendation accuracy

## 📈 Key Insights

- **Melancholy**, **Hopeful**, and **Anxious** were the most frequent complex emotional tones across the movie dataset
- The system shows particularly strong performance in detecting **Sad** emotions (F1-score: 0.9216)
- Hybrid emotion modeling provides more nuanced recommendations compared to traditional single-emotion approaches
- User agency in mood expression leads to higher satisfaction and engagement

## 🔬 Research Contributions

1. **Novel Approach**: First system to utilize user-input hybrid emotional states for movie recommendations
2. **Psychological Accuracy**: Better modeling of real human emotional complexity
3. **Enhanced User Experience**: More emotionally congruent content recommendations
4. **Balanced Performance**: Achieves good accuracy while maintaining usability

## 📝 Future Work

- Integration with real-time biometric data for automatic mood detection
- Expansion to other content domains (music, books, etc.)
- Development of a web-based user interface
- Implementation of feedback loops for continuous learning
- Multi-modal emotion recognition combining text, audio, and visual inputs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- VIT Chennai for providing the research environment
- The open-source community for the excellent NLP libraries
- Movie database providers for the dataset

## 📧 Contact

**Puneet Chandna**  
Email: puneetchandna7@gmail.com  
Institution: VIT Chennai, Tamil Nadu, India

---

*This project represents a significant advancement in affective computing for recommender systems, bridging the gap between human emotional complexity and personalized content delivery.*
