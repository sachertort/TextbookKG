# TextbookKG

This repository contains code for using methods and running experiments and evaluations for the thesis entitled *Construction of Educational Knowledge Graphs and Their Application in Question Answering*. Below you can find the breakdown of modules and instructions for running.

## Concept and Relation Extraction
*(Section 3.1)*

To perform concept and relation extraction from GraphRAG [(Edge et al., 2024)](https://arxiv.org/abs/2404.16130), follow the following pipeline:
1. To do it from scrath, you should create the project root directory, containing `input`, `prompts`, and `settings.yaml`, following [this guide](https://github.com/microsoft/graphrag/blob/main/docs/config/init.md). However, to reproduce our experiments, you can just utilize the `graphrag` directory in this repository, as it contains all the necessary initial components, as well as precomputed `output` (see [the detailed output table schemas](https://github.com/microsoft/graphrag/blob/main/docs/index/outputs.md)).
2. Please note that `graphrag/output/create_final_entities.parquet` contains auto-tuned version of the prompt obtained using the following [guide](https://github.com/microsoft/graphrag/blob/main/docs/prompt_tuning/auto_prompt_tuning.md) and the textbook converted to `.txt` using [`PyPDF2`](https://pypdf2.readthedocs.io/en/3.x/) in `graphrag/input/slp.txt`.
3. Run GraphRAG's Indexer using [this example](https://github.com/microsoft/graphrag/blob/main/docs/get_started.md#running-the-indexing-pipeline) from their documentation.

### Concept Deduplication
*(Section 3.1.1)*

Run concept deduplication pipeline using command:
```
python methods/concept_deduplication.py
```
All the output files are saved to the `postprocessing` directory (they are already there).