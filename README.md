
***
```markdown
# 🎯 Semantic Media Matching Engine

> An AI-powered recommendation system built for the **Fundamentals of AI and ML** course.
> **Author:** Prateek Kumar

This project uses Natural Language Processing (NLP) to move beyond basic keyword searches. By analyzing the contextual "vibe" and semantic meaning of a user's prompt, it recommends the closest matching video game, anime, or movie from a curated database.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [How It Works (Under the Hood)](#-how-it-works-under-the-hood)
3. [Project Structure](#-project-structure)
4. [Getting Started](#-getting-started)
5. [Example Queries](#-example-queries)

---

## ✨ Key Features

* **Context-Aware Search:** Matches intent rather than exact phrasing (e.g., searching for "cursed energy" finds *Jujutsu Kaisen* without needing the exact title).
* **Robust Text Processing:** Utilizes `scikit-learn` to filter out grammatical noise (stop-words) and focus on highly descriptive keywords.
* **Mathematical Precision:** Returns a calculated "Confidence Score" to show exactly how strong the match is.
* **Decoupled Architecture:** Media data is stored externally in a CSV, meaning the database can be updated without ever touching the core Python logic.
* **Error Resiliency:** Built-in Pandas data-cleaning ensures the program won't crash even if the CSV contains empty rows or missing values.

---

## 🧠 How It Works (Under the Hood)

The engine follows a standard Machine Learning pipeline to translate human language into a format the computer can calculate:

1. **TF-IDF Vectorization:** (Term Frequency-Inverse Document Frequency) The engine scans the dataset and assigns mathematical weights to words. Rare, identifying words (like "Witcher" or "Targaryen") are given heavy weight, while common words (like "the" or "and") are ignored.
2. **Vector Space Mapping:** Every movie, game, and anime description is plotted as a coordinate in a multi-dimensional space.
3. **Cosine Similarity Calculation:** When you type a query, your sentence is also plotted in that same space. The engine calculates the angle between your query and the database items using the following formula:

$$similarity = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

The item with the smallest angle (closest to 1.0) is returned as your top recommendation.

---

## 📂 Project Structure

```text
semantic-media-matcher/
│
├── main.py                  # Core NLP logic, TF-IDF vectorizer, and CLI loop
├── media_database.csv       # External dataset containing titles, types, and descriptions
├── requirements.txt         # List of Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed on your system. 

### Installation
1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/semantic-media-matcher.git](https://github.com/YOUR_USERNAME/semantic-media-matcher.git)
   cd semantic-media-matcher
   ```

2. Install the required Machine Learning libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: This installs `pandas` and `scikit-learn`)*

### Execution
Run the main script from your terminal:
```bash
python main.py
```

---

## 📝 Example Queries

Once the engine is running, try typing these prompts to see the NLP in action:

* *"A dark fantasy involving monster hunting."* → **Matches: The Witcher 3**
* *"Sorcerers fighting powerful curses."* → **Matches: Jujutsu Kaisen**
* *"An action RPG with elemental magic."* → **Matches: Genshin Impact**
* *"Noble families fighting for a throne."* → **Matches: Game of Thrones**


```
