citegraph
==============================

Citation Network

# Project Title

## 1. Team Information
- [ ] Team Name: Blue Team
- [ ] Team Members (Name & Email): Alen Koikkara (akoikkar@depaul.edu), Sujay Pookkattuparambil(spookkat@depaul.edu)
- [ ] Course & Section: SE 489

## 2. Project Overview
- [ ] Brief summary of the project (2-3 sentences):
        CiteGraph is a GNN based system for classifying research papers into topics using the structure and content of a citation network.
        It uses node features and edge connections to improve classification performance.
- [ ] Problem statement and motivation:
        The project aims to develop an efficient and accurate system for classifying academic papers based on their citation networks and content.
        By leveraging graph neural networks, we can capture both the structural relationships between papers and their content features.
- [ ] Main objectives:
        - Build a reproducible ML pipeline for classifying nodes in a citation network.
        - Use graph-based models (GCN/GAT) for semi-supervised node classification.
        - Integrate open-source tooling (StellarGraph + MLflow) into a sustainable MLOps workflow.
        - Track experiments and version control collaboratively using Git and Cookiecutter.
        - Implement performance profiling and monitoring for model training and inference.

## 3. Project Architecture Diagram
![MLOps Architecture](reports/figures/architecturemlops.png)

The architecture diagram above illustrates the MLOps workflow of CiteGraph, showing the integration of data processing, model training, and deployment components.

## 4. Phase Deliverables
- [ ] [PHASE1.md](./PHASE1.md): Project Design & Model Development
- [ ] [PHASE2.md](./PHASE2.md): Enhancing ML Operations
- [ ] [PHASE3.md](./PHASE3.md): Continuous ML & Deployment

## 5. Setup Instructions

### Prerequisites
- Python 3.11 or higher
- Git
- Conda (Anaconda or Miniconda)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/citegraph.git
cd citegraph
```

2. Create and activate a conda environment:
```bash
# Create a new conda environment
conda create -n citegraph python=3.11
conda activate citegraph

# Install other dependencies from requirements.txt
pip install -r requirements.txt
```

### Project Dependencies
The project uses the following main dependencies:
- PyTorch (~=2.5)
- PyTorch Geometric (~=2.5)
- NetworkX (~=3.4)
- scikit-learn (~=1.6)
- isort (==6.0.1)
- ruff (==0.11.8)
- mypy (==1.15.0)
- click (==8.1.8)
- python-dotenv (==0.9.9)
- psutil (==5.9.8)
- rich (==13.9)
- hydra-core (~=1.3)

### Running the Code

1. Training the model:
```bash
python src/models/model1/train.py
```

2. Running inference:
```bash
python src/models/model1/inference.py
```

### Performance Profiling

The project includes built-in performance profiling using Python's cProfile. During training, the profiler will:
- Track function call counts and execution times
- Generate a detailed profile report (training_profile.prof)
- Display the top 20 time-consuming functions
- Save profiling results for later analysis

To analyze the profiling results:
```bash
# Using pstats
python -m pstats training_profile.prof

# Using snakeviz (requires installation)
snakeviz training_profile.prof
```

### Development Setup

Run code quality checks:
```bash
# Run isort to sort imports
isort .

# Run ruff for linting
ruff check .

# Run mypy for type checking
mypy .
```

## 6. Contribution Summary
- [ ] Briefly describe each team member's contributions

        Alen: Setting up the workflow, Architecture diagram, Linting, formatting tools setup with git actions, proposal documentation.

        Sujay: Environment, requirements, model pipelines, data versioning, model training and evaluation, proposal documentation.

## 7. References
- [ ] List of datasets, frameworks, and major third-party tools used
        - Python 3.11
        - PyTorch
        - PyTorch Geometric
        - MLflow
        - scikit-learn
        - matplotlib
        - numpy
        - pandas
        - cProfile (for performance profiling)

### Docker Setup

#### GPU Version (with CUDA support)
```bash
# Build and run with GPU support
docker-compose up --build

# Run specific commands
docker-compose run citegraph python src/models/model1/train.py
```

#### CPU Version (without GPU)
```bash
# Build and run without GPU
docker-compose -f docker-compose.cpu.yml up --build

# Run specific commands
docker-compose -f docker-compose.cpu.yml run citegraph python src/models/model1/train.py
```

#### Docker Commands
- Build the image: `docker-compose build`
- Start the container: `docker-compose up`
- Run a specific command: `docker-compose run citegraph <command>`
- Stop the container: `docker-compose down`
- View logs: `docker-compose logs -f`

Project Organization
------------

```
citegraph/
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```


--------
<p><small>Project based on the <a target="_blank" href="https://github.com/Chim-SO/cookiecutter-mlops/">cookiecutter MLOps project template</a>
that is originally based on <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. 
#cookiecuttermlops #cookiecutterdatascience</small></p>