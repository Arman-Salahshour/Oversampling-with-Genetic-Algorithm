import time
import copy
import random
import zipfile
import warnings
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
import xgboost as xgb
import tensorflow as tf
from scipy import interp
from sklearn.svm import SVC
from sklearn import neighbors
from scipy.spatial import distance
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.impute import KNNImputer
from sklearn.metrics import auc as AUC
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB, ComplementNB, GaussianNB
from sklearn.metrics import accuracy_score, recall_score,f1_score,precision_score, roc_auc_score

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 8))
warnings.filterwarnings('ignore')