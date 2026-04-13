# Majorana-MLQT-Project-EthanVu-CMPE188

## Project Title
Majorana-MLQT-Project-EthanVu-CMPE188

## Team Members
- Ethan Vu

## Problem Statement
This project aims to explore the intersection of machine learning and quantum technologies by developing models to analyze and interpret quantum transport data from semiconductor quantum dots. The goal is to demonstrate quantum optimization techniques, understand modern quantum advancements, and apply machine learning concepts to solve real-world quantum physics problems such as thermometry, charge noise characterization, and quantum dot tune-up. By bridging classical machine learning with quantum data, the project seeks to enhance the efficiency and accuracy of quantum device characterization and control.

## Dataset or Data Source
The project utilizes a comprehensive dataset of quantum transport measurements from semiconductor quantum dots, including:
- **Raw Data**: Experimental measurements from charge noise, dot tuneup (A1), and TGP2 tuneup experiments.
- **Converted Data**: Processed H5 files for various quantum dot configurations including cut loops (A/B), injector, MPR (A1/A2/B1), QDMZM (A1), QPP, thermometry at different temperatures (30mK to 302mK), and trivial states (A/B).
- **Simulated Data**: Synthetic datasets for figures (Fig4, FigS3, FigS4, FigS8, FigS11) including QDL, QDR, S12 data, and thermometry references.
- **Mapped Data**: Feature-engineered datasets for specific tasks like charge noise, map classification, QDMZM alignment, sequence readout, TGP2 transport, thermometry, and tuneup, with corresponding CSV files, mapping JSONs, and sample manifests.

Data is stored in HDF5 (.h5) and CSV formats, sourced from quantum physics experiments and simulations.

## Planned Model/System Approach
The system will employ a hybrid approach combining classical machine learning techniques with quantum-inspired algorithms:
- **Data Processing Pipeline**: Custom scripts for H5 to CSV conversion, data parsing, and feature mapping.
- **Machine Learning Models**: Implementation of supervised and unsupervised learning models for tasks such as thermometry prediction, charge noise classification, and quantum dot state identification.
- **Quantum Optimization**: Integration of quantum algorithms (e.g., QAOA or VQE) for optimization problems in quantum data analysis.
- **Evaluation Framework**: Metrics and visualization tools to assess model performance on quantum datasets.
- **User Interface**: A UI component for interactive data exploration and model results visualization.

The approach leverages Python-based libraries (e.g., scikit-learn, TensorFlow/PyTorch) for ML, Qiskit or similar for quantum computing, and HDF5 libraries for data handling.

## Current Implementation Progress
- **Data Management**: Completed H5 to CSV conversion scripts (`h5_to_csv.py`), data parsing mappers (`parsed_mapper.py`), and H5 reading utilities (`read_h5.py`).
- **Data Processing**: Mapped raw and simulated data into structured CSV features for thermometry, charge noise, map classification, QDMZM alignment, sequence readout, TGP2 transport, and tuneup tasks.
- **Model Development**: Initial setup for model training scripts in `src/models/training/`, with evaluation framework in `src/evaluation/`.
- **User Interface**: Basic UI structure in `src/ui/`.
- **Testing**: Basic test script (`test.py`) for validation.
- **Next Steps**: Implement core ML models, integrate quantum algorithms, develop evaluation metrics, and build interactive UI components.

This project is ongoing as part of CMPE 188 coursework and personal exploration into quantum machine learning.
