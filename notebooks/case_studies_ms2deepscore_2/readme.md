# Readme

In this folder you can find the notebooks for creating the case study data for MS2Deepscore 2. 
First spectra are preprocessed by MSDial (not in the notebooks) than the notebooks are run in the following order:
- pre_processing_spectra, the spectra are cleaned and harmonized and identifiers are added.
- Add_annotations, MS2Query annotations are added to make it possible to add structures and compound classes to the visualizations.
- Visualize_embedding_umap, An (interactive) umap visualization is created.
- Create_molecular network, A molecular network is created that can be visualized in cytoscape