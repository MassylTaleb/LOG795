import cv2
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('data/with_pretreatment_span_3_pfe - with_pretreatment_span_3_pfe.csv',dtype=str)
    
    print(df.head())


if __name__ == "__main__":
    main()