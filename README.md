# Authorship Verification of the Disputed Pauline Letters through Deep Learning

This repository contains code for the article:

Beijen, Evy, and Rianne de Heide. 2025. “Authorship Verification of the Disputed Pauline Letters through Deep Learning.” *HIPHIL Novum* 10 (1): 22–39. https://doi.org/10.7146/hn.v10i1.147482.

The study presents a deep learning approach to the authorship verification of the disputed Pauline letters. Specifically, we developed a bidirectional LSTM network to classify segments of the disputed Pauline letters—each approximately 100 words long—as either authored by Paul or not.

When the article was published in 2025, this repository contained two Python files with core functions for the project. These functions were based on the original code used to generate the results in the article, but were adjusted to enhance readability, reusability, and flexibility, with the aim of facilitating the extension and adaptation of our approach in future research. Following a request from the editors of HIPHIL Novum, the repository has been updated in June 2026 to include a full and cleaned version of the code. 

## Files and reproducibility

A full version of the code can be found in the Jupyter Notebook `paul_bilstm.ipynb`. This notebook imports functions from two Python files, also available in this repository:
-	`preprocessing.py`
-	`model_creation_and_hyperparameter_tuning.py`
These files contain core functions to implement the approach outlined in the article. They have been slightly adapted from the Python files published in 2025 to improve clarity, readability, and ease of use in the notebook. 

To run the notebook, a Python environment can be set up using the specifications in `environment.yml`. This environment is intended to approximate the one in which the results reported in the article were originally produced in 2024. When rerunning the notebook in this reconstructed environment in 2026, the plaintext model reproduced the article results. For the lemmatized model, however, exact replication was not achieved, despite using the same code and the same top-level Classical Language Toolkit (CLTK) version number, `cltk==1.2.2`. As explained in the notebook, this discrepancy may be due to changes in CLTK’s underlying resources or dependencies, which may affect lemmatization even when the top-level CLTK version remains the same. Exact reproduction of the lemmatized model should thus be considered unresolved. 

The 2026 run (`run_paul_bilstm.pdf`) of the cleaned notebook is included as a comparison artifact and produced similar, but not identical, results for the lemmatized model as reported in the article. A full snapshot of the environment used for the 2026 rerun is provided in `environment_full.yml` for reference.

More generally, the notebook may not produce the exact results we obtained, despite measures such as setting random seeds. Outcomes depend, for example, on the train-test split, which in turn is influenced by the ordering of text chunks prior to splitting. We recognize the limitation of our article being based on a single run and highlight, as discussed in the article, that the train-test split and other sources of randomness can have an amplified effect given the relatively small size of the dataset. 

The Greek text data used in this project are not included in this repository. Information for obtaining and structuring the text data is provided in the Jupyter Notebook `paul_bilstm.ipynb`. The repository includes the code, environment files, documentation, and a PDF of the 2026 rerun, but not the raw or preprocessed text data.

## Known methodological limitations

This project was exploratory in nature and uses a relatively small dataset. The results may be sensitive to preprocessing choices (e.g., chunking strategy, accent removal), train-test splitting, random seeds, and lemmatization behavior.

Specific limitations include:

1. The tokenizer in the article pipeline is fit on the full subset A before the train-test split. In future research, the tokenizer should preferably be fit only on the training data (see `paul_bilstm.ipynb` for further explanation).
2. The lemmatized model depends on CLTK lemmatization, which can vary across resources or dependency states and might be negatively impacted by preprocessing choices (see `paul_bilstm.ipynb` for further explanation).
3. The main test results are based on chunk-level splitting. Since chunks from the same text may be related, future work might consider letter-level validation or leave-one-letter-out evaluation.
4. We used the held-out set for exploratory error analysis, which may have introduced indirect biases. Final generalization performance should therefore be interpreted cautiously.
5. The article reports a single final run. Future work should average results across multiple random seeds.

## Generative AI disclosure
The code in this repository was developed with assistance from generative AI tools (GPT models), used over the course of the project (2024-2026) for tasks including generating code, refactoring, debugging, and documentation. Outputs were reviewed and often edited by the author(s). The authors take responsibility for the final code, analysis, and documentation.

## Citation

If you use this repository, please cite the article:

Beijen, Evy, and Rianne de Heide. 2025. “Authorship Verification of the Disputed Pauline Letters through Deep Learning.” *HIPHIL Novum* 10 (1): 22–39. https://doi.org/10.7146/hn.v10i1.147482.

A machine-readable citation file is provided in `CITATION.cff`, so that GitHub and other platforms can automatically display citation information for this repository.

## Article reproduction versus future improvements

This repository prioritizes faithful reproduction of the article pipeline. For that reason, several choices are retained even where we would make different choices in a new study.

For future work, we recommend:

- fitting the tokenizer only on the training data;
- comparing multiple preprocessing strategies, especially with regard to chunking, accent removal, and lemmatization (see `paul_bilstm.ipynb` for further explanation);
- saving and reusing fixed train-test split indices;
- averaging results across multiple random seeds;
- evaluating robustness with letter-level or leave-one-letter-out validation;
- storing preprocessed plaintext and lemmatized chunks.

## License

The code and software documentation in this repository are released under the MIT License; see `LICENSE.md`.