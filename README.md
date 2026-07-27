# Resume-Similarity-and-Job-Matching-Using-TF-IDF

This project uses **Natural Language Processing (NLP)** to compare resumes with a job description and identify the resumes that best match the required job skills.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors and **Cosine Similarity** to calculate the similarity between each resume and the job description.

## Features

* Stores multiple resumes in a Pandas DataFrame
* Converts resume and job description text into TF-IDF vectors
* Calculates similarity scores using Cosine Similarity
* Uses a configurable similarity threshold
* Identifies resumes matching the job requirements

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity

## How It Works

### 1. Resume Data

The project contains sample resumes with different skills:

* Data Scientist
* Web Developer
* Data Analyst

### 2. Job Description

The job description requires skills such as:

* Python
* Machine Learning
* SQL
* Data Analysis

### 3. TF-IDF Vectorization

`TfidfVectorizer` converts the text documents into numerical vectors.

```python
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
```

### 4. Cosine Similarity

Cosine similarity compares the job description with each resume:

```python
similarity_scores = cosine_similarity(
    tfidf_matrix[-1],
    tfidf_matrix[:-1]
).flatten()
```

A higher similarity score means the resume has more relevant terms in common with the job description.

### 5. Resume Filtering

A threshold is used to identify matching resumes:

```python
threshold = 0.2

matching_resumes = df[
    df['similarity_score'] >= threshold
]
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/resume-similarity-matching.git
```

Navigate to the project directory:

```bash
cd resume-similarity-matching
```

Install the required libraries:

```bash
pip install pandas scikit-learn
```

## Running the Project

Run the Python script:

```bash
python resume_similarity.py
```

## Example Output

```text
Resume Similarity Scores:

   resume_id  similarity_score
0          1          0.75
1          2          0.00
2          3          0.40
```

The exact similarity scores may vary depending on the dataset and text content.

## Project Structure

```text
resume-similarity-matching/
│
├── resume_similarity.py
└── README.md
```

## Important Note

The similarity score is based on matching words and terms. This project does not fully understand the meaning or context of a resume.

For more advanced resume screening, the project could be improved using:

* Word Embeddings
* BERT
* Sentence Transformers
* Named Entity Recognition (NER)
* Machine Learning Classification Models

## Future Improvements

* Add resume upload functionality
* Support PDF and DOCX resumes
* Create a web interface using Flask or Streamlit
* Rank all resumes from highest to lowest similarity
* Extract skills automatically
* Use advanced NLP models for semantic similarity

## License

This project is open-source and available for educational purposes.
