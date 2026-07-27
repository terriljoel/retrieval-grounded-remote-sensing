# Retrieval-Grounded Remote Sensing

A deep-learning project for detecting and verifying objects in very-high-resolution remote-sensing imagery.

The project starts with a YOLOv8 object-detection baseline using the NWPU VHR-10 dataset. Later stages will use image embeddings, FAISS semantic retrieval, and a vision-language model (VLM) to verify detector predictions.

## Project objectives

1. Train and evaluate an object detector on NWPU VHR-10.
2. Analyse false positives, false negatives, and localisation errors.
3. Generate embeddings for detected objects and reference images.
4. Retrieve similar examples using FAISS.
5. Use a VLM and retrieved evidence to recommend:
   - Accept
   - Reject
   - Relabel
   - Human review
6. Compare:
   - Detector only
   - Detector with same-class retrieval and VLM verification
   - Detector with semantic retrieval and VLM verification

## Dataset

The initial object-detection experiments use **NWPU VHR-10**, containing ten classes:

1. Airplane
2. Ship
3. Storage tank
4. Baseball diamond
5. Tennis court
6. Basketball court
7. Ground track field
8. Harbor
9. Bridge
10. Vehicle

The original dataset is stored in shared Google Drive and is not committed to GitHub.

## Repository structure

```text
retrieval-grounded-remote-sensing/
├── configs/                 # Dataset and experiment configurations
├── docs/                    # Decisions and project documentation
├── notebooks/               # Jupyter and Colab notebooks
├── shared_resources/        # Class mappings and other small shared files
├── src/                     # Reusable Python modules
├── tests/                   # Automated tests
├── .gitignore
├── README.md
└── requirements.txt

shared_resources is meant to be immutable shared folder accessed via Google Drive