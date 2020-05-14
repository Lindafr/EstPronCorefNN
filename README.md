# PronCorefNN
See on osa Linda Freienthali magistritööst "Pronominaalsete viitesuhete automaatne lahendamine eesti keeles närvivõrkude abil".


# PronCorefNN
This is a part of Linda Freienthal's master thesis "Pronominal coreference resolution in Estonian with neural networks".

*do_NNa.py* trains a network called *NNa* with 7-fold method 15 times and outputs the averages of the results. The input data is in file *ccorpusSciKitLearn.txt*. This code outputs:

- *NNa_results.txt* file (containing confusion matrix info, average MCC, loss, accuracy, recall, precision and F1 results on test set), 
- *NNa_confused.png* (containing confusion matrix with percentages), 
- *NNa_loss_val.pdf* (containing average loss values on training and validation data), 
- *NNa_acc_val.pdf* (containing average accuracy values on training and validation data) and 
- *NNa.png* (containing the architecture of the network). 

All other *do_\*.py* files work in similar way. For more information about the networks read the thesis.

These codes work with:

- Keras (2.3.1)
- matplotlib (2.2.5)
- numpy (1.16.6)
- pandas (0.24.2)
- pydot (1.2.1)
- scikit-learn (0.20.4)
- scipy (1.2.3)
- seaborn (0.9.1)
- tensorflow (1.14.0)