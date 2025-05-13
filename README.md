## Files & Instructions to run

- **train.py**: Contains the code for model training, validation, and loss calculation.
- **test.py**: Code for testing the model on the test set and computing BLEU scores.
- **encoder.py**: Implements the encoder block with positional encodings and self-attention layers.
- **decoder.py**: Implements the decoder block with masked self-attention and encoder-decoder attention.
- **utils.py**: Utility functions for data preprocessing, tokenization, padding and more.
- **testbleu.txt**: A text file containing BLEU scores of all sentences in the test set.
- **transformer.pt**: The saved pretrained model file obtained after running `train.py`.

To train the model, run `train.py`. To test the model, run `test.py`.


