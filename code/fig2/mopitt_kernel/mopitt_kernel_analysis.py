import datetime
import pickle

import netCDF4
import pandas as pd
import shapely.vectorized
import xarray as xr
import glob
import numpy as np
from cartopy import crs
from cartopy.feature import NaturalEarthFeature, ShapelyFeature
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from wrf import getvar, ll_to_xy, latlon_coords, get_cartopy, to_np, cartopy_xlim, cartopy_ylim
import cmaps as cs

import cartopy.io.shapereader as shpreader
plt.style.use('science')


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


pres_pfl = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]


def kernel(data, wrfin, out):
    wrfin_lat = wrfin.XLAT.data
    wrfin_lon = wrfin.XLONG.data
    min_lat = wrfin_lat.min() - 1
    max_lat = wrfin_lat.max() + 1
    min_lon = wrfin_lon.min() - 1
    max_lon = wrfin_lon.max() + 1

    frame_filtered = frame.loc[(frame.dofs > 0.5) & (frame.dofs < 3)]
    arr = []
    for i, row in frame_filtered.iterrows():
        ker = row.avgker
        pres = row.surf_pres
        ker = np.where(ker == -999, np.nan, ker)
        if pres > 900:
            arr.append(np.sum(ker, axis=1))
        elif pres > 800:
            ker_tmp = [np.nan]
            ker_tmp.extend(np.sum(ker, axis=1)[1:])
            arr.append(ker_tmp)
        elif pres > 700:
            ker_tmp = [np.nan, np.nan]
            ker_tmp.extend(np.sum(ker, axis=1)[2:])
            arr.append(ker_tmp)
        elif pres > 600:
            ker_tmp = [np.nan, np.nan, np.nan]
            ker_tmp.extend(np.sum(ker, axis=1)[3:])
            arr.append(ker_tmp)
        elif pres > 500:
            ker_tmp = [np.nan, np.nan, np.nan, np.nan]
            ker_tmp.extend(np.sum(ker, axis=1)[4:])
            arr.append(ker_tmp)
        else:
            continue

    ak_pfl = np.nanmean(arr, axis=0)
    ak_std = np.nanstd(arr, axis=0)

    return ak_pfl, ak_std

def draw_ak_pfl(pfl_2017, std_2017, pfl_2018, std_2018, pfl_2019, std_2019, pfl_2020, std_2020):


    fig = plt.figure(figsize=(4, 10))

    # ax_2017 = fig.add_subplot(1, 4, 1)
    # ax_2018 = fig.add_subplot(1, 4, 2)
    # ax_2019 = fig.add_subplot(1, 4, 3)
    # ax_2020 = fig.add_subplot(1, 4, 4)

    x = pres_pfl

    plt.errorbar(pfl_2017, x, xerr=std_2017, label='2017', fmt='.-' , elinewidth=1, capsize=4)
    plt.errorbar(pfl_2018, x, xerr=std_2018, label='2018', fmt='.-', elinewidth=1, capsize=4)
    plt.errorbar(pfl_2019, x, xerr=std_2019, label='2019', fmt='.-', elinewidth=1, capsize=4)
    plt.errorbar(pfl_2020, x, xerr=std_2020, label='2020', fmt='.-', elinewidth=1, capsize=4)

    plt.xlabel('CO Averaging Kernel', {'size': 16})
    plt.ylabel('Pressure (hPa)', {'size': 16})
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig('ak.png', dpi=300)


if __name__ == '__main__':
    # mop_pkl_2017 = r'H:\code\wrf\co_eu_eval\revision\mopitt_data\mopitt_data_2017'
    # mop_pkl_2018 = r'H:\code\wrf\co_eu_eval\revision\mopitt_data\mopitt_data_2018'
    # mop_pkl_2019 = r'H:\code\wrf\co_eu_eval\revision\mopitt_data\mopitt_data_2019'
    # mop_pkl_2020 = r'H:\code\wrf\co_eu_eval\revision\mopitt_data\mopitt_data_2020'
    #
    # wrfin = xr.open_dataset('wrfinput_d01')
    #
    # with open(mop_pkl_2017, 'rb') as fp:
    #     frame = pickle.load(fp)
    #
    # pfl_2017, std_2017 = kernel(frame, wrfin, '')
    #
    # with open(mop_pkl_2018, 'rb') as fp:
    #     frame = pickle.load(fp)
    #
    # pfl_2018, std_2018 = kernel(frame, wrfin, '')
    #
    # with open(mop_pkl_2019, 'rb') as fp:
    #     frame = pickle.load(fp)
    # pfl_2019, std_2019 = kernel(frame, wrfin, '')
    #
    # with open(mop_pkl_2020, 'rb') as fp:
    #     frame = pickle.load(fp)
    #
    # pfl_2020, std_2020 = kernel(frame, wrfin, '')

    with open('ak_pfls', 'rb') as fp:
        arr = pickle.load(fp)
        # pfl_2017 = arr[0], std_2017, pfl_2018, std_2018, pfl_2019, std_2019, pfl_2020, std_2020

    # with open('ak_pfls', 'wb') as fp:
    #     pickle.dump([pfl_2017, std_2017, pfl_2018, std_2018, pfl_2019, std_2019, pfl_2020, std_2020], fp)

    draw_ak_pfl( arr[0],  arr[1],  arr[2],  arr[3],  arr[4],  arr[5],  arr[6],  arr[7])
