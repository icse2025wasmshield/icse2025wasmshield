# %env PYTORCH_ENABLE_MPS_FALLBACK=1

import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')
plt.rcParams["axes.grid"] = False

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image

from wasmshield.preprocessing import preprocess_image

def draw_files(files, size = 128, group_mapping = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 1, 9: 2, 10: 1, 11: 2, 12: 2,}, figsize=(20, 10), cols=[], rows=[], use_pil=False, compress=False):
    
    nb_channels = len(set(group_mapping.values()))
    fig, axes = plt.subplots(nrows=nb_channels, ncols=len(files), figsize=figsize, facecolor='white')
    if len(cols)==0:
        cols = [x.split('/')[-1] for x in files]
    if len(rows)==0:
        rows = [f'Channel {x}' for x in range(1,nb_channels+1)]

    pad = 5
    if nb_channels>1:
        for ax in (axes.flatten()):
            ax.grid(False)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        for ax, col in zip(axes[0], cols):
            ax.grid(False)
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        for ax, row in zip(axes[:,0], rows):
            ax.grid(False)
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
        
    for idx,file in enumerate(files):
        img = preprocess_image(file, size, group_mapping=group_mapping, use_pil=use_pil, compress=compress)
        for i,channel in enumerate(img):
            plt.grid(None)
            plt.subplot(3,len(files),(i)*len(files)+1+idx)
            plt.grid(None)
            channel = Image.fromarray((
                        channel #* 255
                    ).astype(np.uint8), mode='L')
            plt.imshow(channel, interpolation='nearest', cmap='gray')
            plt.grid(None)
            if nb_channels == 1:
                plt.title(cols[idx])
            plt.grid(None)
        
    plt.show()

def draw_files_horizontal(files, size = 128, group_mapping = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 1, 9: 2, 10: 1, 11: 2, 12: 2,}, figsize=(20, 10), rows=[], cols=[], title='Plot', facecolor='white', use_pil=False, compress=False):
    
    nb_channels = len(set(group_mapping.values()))
    fig, axes = plt.subplots(nrows=len(files), ncols=nb_channels, figsize=figsize, facecolor=facecolor)
    fig.tight_layout(pad=0)
    if len(rows)==0:
        rows = [x.split('/')[-1] for x in files]
    if len(rows)==0:
        cols = [f'Channel {x}' for x in range(1,nb_channels+1)]

    pad = 5
    if nb_channels>1:
        for ax in (axes.flatten()):
            ax.grid(False)
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
        for ax, col in zip(axes[0], cols):
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        for ax, row in zip(axes[:,0], rows):   
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)
    
    for idx,file in enumerate(files):
        img = preprocess_image(file, size, group_mapping=group_mapping, use_pil=use_pil, compress=compress)
        for i,channel in enumerate(img):
            ax = axes.flatten()[(idx)*nb_channels+i]
            channel = Image.fromarray((
                        channel #* 255
                    ).astype(np.uint8), mode='L')
            ax.imshow(channel, interpolation='nearest', cmap='gray')
            ax.grid(None)
    fig.show()

def draw_images(images, size = 128, group_mapping = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 2, 5: 2, 6: 2, 7: 0, 8: 1, 9: 2, 10: 1, 11: 2, 12: 2,}, figsize=(20, 10)):
    
    nb_channels = len(set(group_mapping.values()))
    fig, axes = plt.subplots(nrows=nb_channels, ncols=len(images), figsize=figsize)
    cols = [f'Channel {x}' for x in range(1,len(images)+1)]
    rows = [f'Image {x}' for x in range(1,nb_channels+1)]

    pad = 5
    if nb_channels>1:
        for ax, col in zip(axes[0], cols):
            ax.grid(False)
            ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        for ax, row in zip(axes[:,0], rows):
            ax.grid(False)
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
        
    for idx,img in enumerate(images):

        for i,channel in enumerate(img):
            plt.subplot(3,len(images),(i)*len(images)+1+idx)
            channel = Image.fromarray((channel * 255).astype(np.uint8), mode='L')
            plt.imshow(channel, interpolation='nearest', cmap='gray')
            plt.grid(None)
            if nb_channels == 1:
                plt.title(cols[idx])
    plt.show()
    