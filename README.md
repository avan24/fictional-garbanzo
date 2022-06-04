import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import numpy as np
from IPython.core.display import display, HTML

from keras.models import Sequential
from keras.layers import Dense

# reading data
try:
    df = pd.read_csv('symptomData2.csv')
    dfModifiers = pd.read_csv('modifierData3.csv')
    dfTestTreatment = pd.read_csv('testData2.csv')

    dfTest = pd.read_csv('symptomData2Test.csv')
    dfModifiersTest = pd.read_csv('modifierData3Test.csv')
    dfTestTreatmentTest = pd.read_csv('testData2Test.csv')
except:
    print("""
      Dataset not found in your computer.
      """)
    quit()

# Preprocessing data, reading data from data files
OriginalX_train= df
OriginalX_test = dfTest

dataWithSymptomDuration = OriginalX_train.groupby(["patient","outcome"]).agg(lambda x: x[x > 0].count()).reset_index().rename(columns=lambda x: x+ '_severity' if ((x != 'outcome') & (x != 'patient')) else x)
dataWithSymptomSeverity = OriginalX_train.groupby(["patient","outcome"]).agg(lambda x: x[x > 0].count()).reset_index().rename(columns=lambda x: x+'_duration' if ((x != 'outcome') & (x != 'patient')) else x)

dataWithSymptomDurationTest = OriginalX_test.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index().rename(columns=lambda x: x+ '_severity' if ((x != 'outcome') & (x != 'patient')) else x)
dataWithSymptomSeverityTest = OriginalX_test.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index().rename(columns=lambda x: x+'_duration' if ((x != 'outcome') & (x != 'patient')) else x)

dataWithModifiers = dfTestTreatment.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index()
dataWithModifiersTest = dfModifiersTest.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index()

dataWithLabTestTreatment = dfModifiers.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index()
dataWithLabTestTreatmentTest = dfTestTreatmentTest.groupby(["patient","outcome"]).agg(lambda x: x[x >= 0].max()).reset_index()

print("Count - Row = " + str(len(dataWithSymptomDuration.index)))
print(dataWithSymptomDuration.head())

print("Severity - Row = " + str(len(dataWithSymptomSeverity.index)))
print(dataWithSymptomSeverity.head())

print("data with modifiers - Row = " + str(len(dataWithModifiers.index)))
print(dataWithModifiers.head())

print("data with test treatment - Row = " + str(len(dataWithLabTestTreatment.index)))
print(dataWithLabTestTreatment.head())

#combine train data
dataWithSymptomSeverity = dataWithSymptomSeverity.loc[:, dataWithSymptomSeverity.columns != 'outcome']
dataWithModifiers = dataWithModifiers.loc[:, dataWithModifiers.columns != 'outcome']
dataWithLabTestTreatment = dataWithLabTestTreatment.loc[:, dataWithLabTestTreatment.columns != 'outcome']

combinedSymptomSeverityDuration = pd.concat([dataWithSymptomDuration.set_index('patient'), dataWithSymptomSeverity.set_index('patient'), dataWithModifiers.set_index('patient'), dataWithLabTestTreatment.set_index('patient')], axis=1, join='inner')

print("Combine - Row = " + str(len(combinedSymptomSeverityDuration.index)))
print(combinedSymptomSeverityDuration.head())

#combine test data
dataWithSymptomSeverityTest = dataWithSymptomSeverityTest.loc[:, dataWithSymptomSeverityTest.columns != 'outcome']
dataWithModifiersTest = dataWithModifiersTest.loc[:, dataWithModifiersTest.columns != 'outcome']
dataWithLabTestTreatmentTest = dataWithLabTestTreatmentTest.loc[:, dataWithLabTestTreatmentTest.columns != 'outcome']

combinedSymptomSeverityDurationTest = pd.concat([dataWithSymptomDurationTest.set_index('patient'), dataWithSymptomSeverityTest.set_index('patient'), dataWithModifiersTest.set_index('patient'), dataWithLabTestTreatmentTest.set_index('patient')], axis=1, join='inner')
print("Combine Test - Row = " + str(len(combinedSymptomSeverityDurationTest.index)))
print(combinedSymptomSeverityDurationTest.head())

# data ready for run
totalData = combinedSymptomSeverityDuration.loc[:, combinedSymptomSeverityDuration.columns != 'patient']
totalDataTest = combinedSymptomSeverityDurationTest.loc[:, combinedSymptomSeverityDurationTest.columns != 'patient']

print("Count Train data = ", str(len(totalData.index)), "Test data = ", str(len(totalDataTest.index)))

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(totalData, label="outcome")
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(totalDataTest, label="outcome")

# --- List All features
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))

# Train the model
modelSymtomSeverityDuration = tfdf.keras.RandomForestModel()
modelSymtomSeverityDuration.fit(train_ds)

# Summary of the model structure.
modelSymtomSeverityDuration.summary()

# Evaluate the model.
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(combinedSymptomSeverityDurationTest, label="outcome")
modelSymtomSeverityDuration.compile(metrics=["accuracy"])

print("----------- evaluating test data ---------- ")
print(modelSymtomSeverityDuration.evaluate(test_ds))

# Inspect the result
print("----------- Variable Importances ---------- ")
print(modelSymtomSeverityDuration.make_inspector().variable_importances())

print("----------- Features ---------- ")
print(modelSymtomSeverityDuration.make_inspector().features())

print("--- prediction---")
print(modelSymtomSeverityDuration.predict(test_ds))

print("----------- Generate graph ---------- ")
html = tfdf.model_plotter.plot_model_in_colab(modelSymtomSeverityDuration, tree_idx=0, max_depth=20)
with open('model_combine.html', 'w') as f:
    f.write(html.data)

