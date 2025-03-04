# Document Parser Evaluator

A comprehensive tool for evaluating the quality and coverage of question-answer pairs extracted from text documents.

## Overview

This repository contains a Python toolkit for assessing how well a set of question-answer pairs captures the key information from a source document. The evaluator uses natural language processing techniques to analyze information coverage, question quality, and other metrics to provide a comprehensive assessment of document parsing performance.

## Features

- **Coverage Analysis**: Measures how well QA pairs cover key sentences, named entities, and important content words from the document
- **QA Quality Evaluation**: Assesses question formatting, length, and diversity
- **Automated Key Information Detection**: Algorithmically identifies the most important sentences in a document
- **Question Type Analysis**: Categorizes and analyzes the distribution of different question types (What, When, Why, How, etc.)
- **Visualization**: Optional visualization of sentence importance distribution with identified optimal cutoff points
- **Comprehensive Metrics**: Generates detailed performance metrics with an overall quality score

## Installation

### Prerequisites

- Python 3.7+
- Required Python packages:
  - nltk
  - scikit-learn
  - numpy
  - pandas
  - spacy
  - matplotlib
  - kneed (for elbow/knee detection)

### Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/document-parser-evaluator.git
   cd document-parser-evaluator
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download required NLTK and Spacy resources:
   ```
   python -m nltk.downloader punkt stopwords
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

```bash
python eval.py document.txt qa_pairs.json
```

Where:
- `document.txt` is the input document text file
- `qa_pairs.json` is a JSON file containing question-answer pairs in the format:
  ```json
  [
    {"question": "What is X?", "answer": "X is Y."},
    {"question": "Who did Z?", "answer": "Person A did Z."}
  ]
  ```

### Advanced Options

```bash
python eval.py document.txt qa_pairs.json --visualize --output-dir ./results --output-prefix example --verbose
```

Available options:
- `--visualize`: Generate visualization of sentence importance distribution
- `--output-dir`: Directory to save output files (default: current directory)
- `--output-prefix`: Prefix for output filenames
- `--no-spacy`: Disable Spacy NER (faster but less detailed)
- `--verbose`: Print detailed evaluation results to console

### Using the Helper Script

For batch processing, you can use the provided shell script:

```bash
./run.sh
```

## Repository Structure

```
├── README.md                # This documentation
├── eval.py                  # Main evaluation script
├── inputs/                  # Example input documents and QA pairs
│   ├── long_paper.txt       # Sample long paper
│   ├── long_paper.json      # QA pairs for long paper
│   ├── short_report.txt     # Sample short report
│   └── ...
├── outputs/                 # Evaluation results
│   └── ...
├── run.sh                   # Helper script for batch evaluation
├── test.ipynb               # Jupyter notebook for testing and examples
└── utils.py                 # Utility functions
```

## Evaluation Metrics

The evaluator calculates the following metrics:

1. **Coverage Metrics**:
   - Sentence Coverage: How well QA pairs cover key sentences
   - Entity Coverage: Percentage of named entities included in QA pairs
   - Word Coverage: Percentage of important content words covered

2. **Quality Metrics**:
   - Proper Question Format: Percentage of questions starting with appropriate question words
   - Question Mark Usage: Percentage of questions ending with a question mark
   - Average Question Length: Mean number of words in questions
   - Average Answer Length: Mean number of words in answers

3. **Question Type Analysis**:
   - Distribution of question types (What, Where, When, Who, Why, How, Which, Other)

4. **Summary Statistics**:
   - Total QA Pairs
   - Document Length (words)
   - QA Pairs Per Sentence
   - Key Sentences Used

## Output Format

The evaluator produces a JSON file with detailed results:

```json
{
  "overall_score": 85.7,
  "coverage_metrics": {
    "sentence_coverage": 0.76,
    "entity_coverage": 0.89,
    "word_coverage": 0.65,
    "num_qa_pairs": 15,
    "doc_length": 1250,
    "qa_density": 0.32,
    "key_sentences_extracted": 12
  },
  "quality_metrics": {
    "avg_question_length": 8.5,
    "avg_answer_length": 15.2,
    "proper_question_format": 0.93,
    "has_question_mark": 1.0
  },
  "question_type_analysis": {
    "what": 40.0,
    "where": 13.3,
    "when": 6.7,
    "who": 20.0,
    "why": 6.7,
    "how": 13.3,
    "which": 0.0,
    "other": 0.0
  },
  "summary": {
    "total_qa_pairs": 15,
    "document_length_words": 1250,
    "qa_pairs_per_sentence": 0.32,
    "key_sentences_used": 12
  }
}
```

## How It Works

The `TextParserEvaluator` class employs several NLP techniques:

1. **Key Sentence Extraction**: Uses TF-IDF scores to identify the most important sentences
2. **Optimal Sentence Selection**: Employs the elbow/knee method to determine how many key sentences to extract
3. **Named Entity Recognition**: Uses Spacy to identify important entities in the document
4. **Coverage Calculation**: Measures how well QA pairs cover key information using cosine similarity and text matching
5. **Question Analysis**: Evaluates question formatting and categorizes question types

## Example

```python
from text_parser_evaluator import TextParserEvaluator

# Initialize evaluator
evaluator = TextParserEvaluator()

# Read document and QA pairs
with open("document.txt", "r") as f:
    document = f.read()

qa_pairs = [
    ("What is the main focus of the paper?", "The paper focuses on climate change."),
    ("Who conducted the research?", "Dr. Smith and his team conducted the research.")
]

# Evaluate
results = evaluator.evaluate(document, qa_pairs)
print(f"Overall Score: {results['overall_score']:.2f}%")
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.