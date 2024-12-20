# Pytorch for NLP

To run the Code you need to have torch installed. I added a make command for this:

```bash
make install requirements-pytorch
```

Otherwise you can install it using pip:

```bash
pip install torch
```

Afterwards you can execute the Code in the Jupyter Notebook in "notebooks/pytorch_for_nlp.ipynb", which uses the modules from this package.

If you wanna train the model you have to set the variable `TRAIN_MODEL` to `True` in the notebook. Otherwise it will just load the pretrained model, or throw an error if it doesn't exist.
