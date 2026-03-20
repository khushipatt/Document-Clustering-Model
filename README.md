# Document Clustering — NLP Sem 6 Project

This is a web app I built for my Sem 6 NLP project. The idea is pretty simple — you give it a pile of text documents and it figures out which ones are talking about similar things, then groups them together automatically. No manual labeling needed.

---

## What's the point of this?

Say you have 500 news articles sitting in a CSV. You don't want to read through all of them just to figure out which ones are about sports, which are about tech, and which are about health. This app does that for you in seconds.

You just upload your file, pick an algorithm, hit run — and it shows you the groups along with charts, word clouds, and the keywords that define each group.

---

## Running it locally

**1. Install the packages**
```bash
pip install -r requirements.txt
```

**2. Start the app**
```bash
streamlit run app.py
```

**3. Open your browser** at `http://localhost:8501` and you're good to go.

---

## What's in the repo

```
├── app.py            # The Streamlit UI — everything the user sees
├── clustering.py     # Where all the actual ML stuff happens
├── requirements.txt  # Python packages you'll need
├── config.toml       # Just some theme/color settings for Streamlit
└── sample_data.csv   # A small dataset you can use to try it out
```

---

## How it actually works

### Step 1 — You upload your data
Either a CSV (pick which column has your text) or a plain TXT file where each paragraph is treated as one document. There's also sample data built in if you just want to see it in action — it covers Tech, Sports, Health, and Finance topics.

### Step 2 — The text gets cleaned up
Before anything ML-related happens, the raw text goes through some basic cleaning. Emails, URLs, and random symbols get stripped out, everything gets lowercased, extra spaces are removed. Nothing fancy, just making sure the data is consistent before feeding it to the model.

### Step 3 — Words get turned into numbers (TF-IDF)
Machine learning models can't read words — they work with numbers. So we use **TF-IDF** (Term Frequency–Inverse Document Frequency) to convert each document into a list of numbers. Basically, it gives every word a score based on how often it shows up in that document versus how common it is across all documents.

> Words like "the" or "is" that appear everywhere get a low score. Words like "photosynthesis" that only appear in specific documents get a high score. That's how the model learns what each document is actually about.

### Step 4 — Clustering (pick one of three algorithms)

| Algorithm | What it does | When to use it |
|---|---|---|
| **K-Means** | Picks K center points and groups documents around the nearest one, keeps adjusting until the groups stabilize | Good default choice, fast on large datasets |
| **Hierarchical** | Starts by treating each document as its own group, then keeps merging the two closest ones | Better for smaller datasets, gives a cleaner sense of how groups relate |
| **LDA (Topic Modeling)** | Instead of hard groups, it assumes each document is a blend of topics and finds that mix | Great when documents overlap multiple themes |

### Step 5 — You get a bunch of visuals

Once it's done clustering, the app shows you:

- **Scatter Plot** — every document as a dot on a 2D chart, colored by which cluster it landed in
- **Bar + Pie Chart** — a quick look at how many documents are in each group
- **Word Clouds** — the most important words for each cluster, shown visually
- **Elbow Chart** *(K-Means only)* — helps you figure out the right number of clusters if you're unsure
- **Keywords Table** — top words per cluster in a clean table you can scan quickly
- **Document Explorer** — click into any cluster and read the actual documents inside it
- **Download buttons** — grab the cluster assignments or keywords as a CSV

---

## What you can configure

Everything's in the sidebar on the left:

- **Number of Clusters** — how many groups you want (anywhere from 2 to 12)
- **Algorithm** — K-Means, Hierarchical, or LDA
- **Linkage Type** *(only shows up for Hierarchical)* — Ward, Average, Complete, or Single
- **Max TF-IDF Features** — how big the vocabulary should be
- **Top Words per Cluster** — how many keywords to show per group
- **Max Documents** — useful if your dataset is huge and you want results faster

---

## Is the clustering any good?

The app shows a **Silhouette Score** after every run. It's a number between -1 and 1 that tells you how well-separated the clusters are.

- Near **+1** means each cluster is tight and clearly separated from the others — that's what you want
- Near **0** means the clusters are kind of bleeding into each other
- Near **-1** means something probably went wrong

For K-Means specifically, the **Elbow Method** chart plots inertia vs number of clusters. You look for the "elbow" — the point where adding more clusters stops making a big difference. That's usually the sweet spot for K.

---

## Libraries used

| Library | What it's doing here |
|---|---|
| `streamlit` | Building the whole web UI |
| `scikit-learn` | K-Means, Hierarchical clustering, LDA, TF-IDF, PCA, Silhouette score |
| `nltk` | Getting the list of stopwords to filter out |
| `plotly` | The interactive charts |
| `matplotlib` | Rendering the word clouds |
| `wordcloud` | Actually generating the word clouds |
| `pandas / numpy` | Handling the data |
| `scipy` | Supporting the hierarchical clustering math |

*Semester 6 NLP project 🎓*
