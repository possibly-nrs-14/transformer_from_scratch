## Files & Instructions to run

- **train.py**: Contains the code for model training, validation, and loss calculation.
- **test.py**: Code for testing the model on the test set and computing BLEU scores.
- **encoder.py**: Implements the encoder block with positional encodings and self-attention layers.
- **decoder.py**: Implements the decoder block with masked self-attention and encoder-decoder attention.
- **utils.py**: Utility functions for data preprocessing, tokenization, padding and more.
- **report.pdf**: A detailed description of the model performance and hyperparameters used.
- **testbleu.txt**: A text file containing BLEU scores of all sentences in the test set.
- **transformer.pt**: The saved pretrained model file obtained after running `train.py`.

To train the model, run `train.py`. To test the model, run `test.py`.

Note: Due to the large size of `transformer.pt`, it could not be included directly in the submission and can be accessed at:
https://drive.google.com/drive/folders/12l8C1mONEsJkL3OJDlvA_R1WxJDekeIG?usp=sharing
