# Authorship Verification of the Disputed Pauline Letters through Deep Learning

This repository contains the core functions of the project "Authorship Verification of the Disputed Pauline Letters through Deep Learning" by Evy Beijen and Rianne de Heide, which was accepted for publication in *HIPHIL Novum*. The study presents a deep learning approach to the authorship verification of the disputed Pauline letters. Specifically, we developed a bidirectional LSTM network to classify segments of the disputed Pauline letters—each approximately 100 words long—as either authored by Paul or not.

## Overview
This repository includes two primary files that reflect the methodology described in the article:

- **`preprocessing.py`**
- **`model_construction_and_evaluation.py`**

These files contain functions that implement the article's approach, including preprocessing of the text data (i.e., letters in Ancient Greek), model construction, and model evaluation. The functions are based on the code used to generate the results in the article but have been adjusted to enhance readability, reusability, and flexibility. We hope this will facilitate the extension and adaptation of our approach in future research.

## Citation

Beijen, Evy, and Rianne de Heide. "Authorship Verification of the Disputed Pauline Letters through Deep Learning." *HIPHIL Novum*, accepted for publication.

## License
This project is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) License.  
For full details, see the [LICENSE](./LICENSE) file or visit the official license page: [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).
