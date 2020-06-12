Chicage taxi data set.

*  Simple data set contains all the data for training and evaluation.
*  Unlabelled data set is a similar simple data set without tip column.
   TODO(b/131873699): Update the unlabelled data set with new data from origin.
*  Big tipper data set contains a new column 'big_tipper' that indicates if tips
   was > 20% of the fare.
   TODO(b/157064428): Remove after label transformation is supported for Keras.
