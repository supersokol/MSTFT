# MSTFT Demo: MetaScaffold TaxoFrame Template for Taxonomy Construction

This repository presents a simple demonstrative implementation of the MetaScaffold TaxoFrame Template (MSTFT) methodology—a novel approach for taxonomy construction using Large Language Models (LLMs). Designed for the NLDB conference, this demo illustrates how structured meta-prompts, dynamic graph-based ranking, and iterative refinement can be integrated to produce consistent, adaptable taxonomies.

## Overview

MSTFT provides a standardized framework that:
- **Integrates Meta-Prompting:** Uses meta-level instructions to guide LLM-based reasoning.
- **Utilizes Dynamic Graphs:** Implements a meta-prompts graph for ranking and retrieving hierarchical taxonomic elements.
- **Ensures Flexibility & Consistency:** Adapts to new data and restructuring needs, while maintaining semantic validation across taxonomic levels.

This demo repository is a condensed example of the approach detailed in the accompanying article.

## Repository Structure

```
MSTFT-Demo/
├── data/
│   └── meta_structure.json       # JSON file containing meta scaffold and taxonomy data.
├── src/
│   ├── langgraph_logic.py        # Core logic for processing taxonomy elements and meta-prompts.
│   └── MSTFTdemo.py              # Demonstration script integrating metadata, graph retrieval, and taxonomy construction.
├── README.md                     # This document.
└── LICENSE                       # Licensing information.
```

- **data/meta_structure.json:** Contains the metadata and hierarchical configuration used by MSTFT. It defines the overall structure, classification criteria, and meta-prompts for taxonomy construction.
- **src/langgraph_logic.py:** Implements core functions for managing taxonomy elements, including creating, validating, and traversing taxonomic concepts.
- **src/MSTFTdemo.py:** A runnable demo that shows how MSTFT integrates meta-prompts, applies dynamic ranking on taxonomy data, and retrieves hierarchical classifications. It also demonstrates index building and retrieval using a cosine similarity function.

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`)
  - numpy
  - langchain (and other related dependencies)
  - Your custom MSTFT modules (included or referenced in the repository)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/supersokol/MSTFT.git
   cd MSTFT-Demo
   ```
2. (Optional) Set up a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the Demo

Execute the demonstration script:
```bash
python src/MSTFTdemo.py
```
This script will:
- Load the meta scaffold from `meta_structure.json`
- Build indices for taxonomy data and meta-prompts
- Perform retrieval based on cosine similarity to simulate dynamic taxonomy ranking
- Print out the constructed taxonomy information and retrieval results

## Customization

You can adjust:
- **Meta-Scaffold Data:** Modify `meta_structure.json` to reflect your domain-specific taxonomy or to experiment with alternative hierarchical configurations.
- **Graph and Retrieval Logic:** Tweak `langgraph_logic.py` functions to refine how meta-prompts are combined and how the ranking algorithm influences taxonomy construction.
- **LLM Settings:** In `MSTFTdemo.py`, experiment with different model parameters (temperature, top_p, penalties) to explore various outputs in concept generation and refinement.

## Contribution & Further Development

Contributions and enhancements to the MSTFT methodology are welcome. This demo serves as a starting point for researchers and practitioners interested in integrating LLM-driven taxonomy construction into broader AI and knowledge organization systems.

For further details on the underlying methodology, please refer to the accompanying article provided in the repository.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to modify and expand this README to better suit any additional details or changes in your approach.
