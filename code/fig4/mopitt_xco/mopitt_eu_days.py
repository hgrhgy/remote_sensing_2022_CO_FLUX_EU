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


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


pres_pfl = [850, 750, 650, 550, 450, 350, 250, 150, 50]



def pfl_to_col(arr_var, surf_co, arr_pres, surf):
    h = []

    lvls = arr_var.shape[0]
    var = []
    pres = []
    if surf > 900:
        var.append(surf_co)
        var.extend([v for v in arr_var])
        pres.append( surf / 2 + 450)
        pres.extend([v for v in arr_pres])
        lvls = arr_var.shape[0] + 1
    elif surf > 800:
        var.append(surf_co)
        var.extend([v for v in arr_var[1:]])
        pres.append(surf / 2 + 400)
        pres.extend([v for v in arr_pres[1:]])
        lvls = arr_var.shape[0]
    elif surf > 700:
        var.append(surf_co)
        var.extend([v for v in arr_var[2:]])
        pres.append(surf / 2 + 350)
        pres.extend([v for v in arr_pres[2:]])
        lvls = arr_var.shape[0] - 1
    elif surf > 600:
        var.append(surf_co)
        var.extend([v for v in arr_var[3:]])
        pres.append(surf / 2 +300)
        pres.extend([v for v in arr_pres[3:]])
        lvls = arr_var.shape[0] - 2
    elif surf > 500:
        var.append(surf_co / 2 + 250)
        var.extend([v for v in arr_var[4:]])
        pres.append(surf)
        pres.extend([v for v in arr_pres[4:]])
        lvls = arr_var.shape[0] - 3
    else:
        lvls = 0
        print('error')


    for lvl in range(lvls):
        try:
            if lvl != 0 and lvl < lvls - 2:
                p_i = pres[lvl]
                p_ip1 = pres[lvl + 1]
                p_im1 = pres[lvl - 1]
                hi = np.abs(-p_i + (p_ip1 - p_i) / np.log(p_ip1 / p_i) + p_i - (p_i - p_im1) / np.log(p_i / p_im1)) * (
                            1 / surf)
                h.append(hi)
            elif lvl == 0:
                p_i = pres[lvl]
                p_ip1 = pres[lvl + 1]
                hi = np.abs(-p_i + (p_ip1 - p_i) / np.log(p_ip1 / p_i)) * (1 / surf)
                h.append(hi)
            else:
                p_i = pres[lvl]
                p_im1 = pres[lvl - 1]
                hi = np.abs(p_i - (p_i - p_im1) / np.log(p_i / p_im1)) * (1 / surf)
                h.append(hi)
        except RuntimeWarning:
            print(p_i)
            print(p_im1)
            print(p_ip1)
            pass

    h = np.array(h)

    var_mean = np.sum(h.T * var)

    return var_mean


def plot(frame, wrfin, out, vmin=100, vmax=300):
    slp = getvar(wrfin, 'slp')
    data = np.zeros(slp.shape)
    cnt = np.zeros(slp.shape)

    frame_filtered = frame
    for i, row in frame_filtered.iterrows():
        x, y = ll_to_xy(wrfin, row.latitude, row.longitude)

        if x >= data.shape[1] or y >= data.shape[0]:
            continue

        data[y, x] += row.surf_co[0]
        cnt[y, x] += 1

    data = data / cnt
    data = np.array(data)

    lats, lons = latlon_coords(slp)

    cart_proj = get_cartopy(slp)

    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                 facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)

    norm = BoundaryNorm(np.arange(vmin, vmax, 20), ncolors=cs.WhiteBlueGreenYellowRed.N, clip=True)

    data[data>vmax]=vmax
    data[data<vmin]=vmin

    ret = plt.pcolormesh(to_np(lons), to_np(lats), data,
                         transform=crs.PlateCarree(),
                         cmap=cs.WhiteBlueGreenYellowRed,
                         vmin=vmin,
                         vmax=vmax)

    cb = fig.colorbar(ret)
    cb.set_label(label='ppb', fontsize=18, family='Times New Roman')
    cb.set_ticks(np.arange(vmin, vmax, 1))
    cb.ax.tick_params(labelsize=16)
    cb.set_ticklabels(np.arange(vmin, vmax, 20))

    plt.savefig(out, dpi=300)

    plt.close()


def plot_latlon(frame, wrfin, out, vmin=60, vmax=120):
    wrfin_lat = wrfin.XLAT.data
    wrfin_lon = wrfin.XLONG.data
    min_lat = wrfin_lat.min() - 1
    max_lat = wrfin_lat.max() + 1
    min_lon = wrfin_lon.min() - 1
    max_lon = wrfin_lon.max() + 1


    frame_filtered = frame.loc[(frame.dofs > 0.5) & (frame.dofs < 3)].groupby(['year','month','day'])



    data_arr = []
    cnt_arr = []

    for n, g in frame_filtered:
        data = np.zeros((int((max_lat - min_lat) * 4) + 1, int((max_lon - min_lon) * 4) + 1))
        cnt = np.zeros((int((max_lat - min_lat) * 4) + 1, int((max_lon - min_lon) * 4) + 1))
        for i, row in g.iterrows():
            lat = row.latitude
            lon = row.longitude

            y = -1
            x = -1
            if min_lat < lat < max_lat:
                x = int((lat - min_lat) * 4)

            if min_lon < lon < max_lon:
                y = int((lon - min_lon) * 4)

            if x == -1 or y == -1:
                continue

            surf_co = row.surf_co[0]
            co_pfl = row.co_pfl[:,0]
            surf_pres = row.surf_pres

            data[x, y] += pfl_to_col(co_pfl, surf_co, pres_pfl, surf_pres)
            cnt[x, y] += 1

        data_arr.append(data)
        cnt_arr.append(cnt)

    data_arr = np.array(data_arr)
    cnt_arr = np.array(cnt_arr)

    with open('%s_data.pkl' % out, 'wb') as fp:
        pickle.dump(data_arr,fp)

    with open('%s_cnt.pkl' % out, 'wb') as fp:
        pickle.dump(cnt_arr,fp)

    lats = np.arange(min_lat, max_lat, 0.25)
    lons = np.arange(min_lon, max_lon, 0.25)

    data = np.sum(data_arr, axis=0) / np.sum(cnt_arr, axis=0)

    cart_proj = crs.PlateCarree()

    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    # Set the GeoAxes to the projection used by WRF
    ax = plt.axes(projection=cart_proj)
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                 facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax.add_feature(states, linewidth=.5, edgecolor="black")
    ax.coastlines('50m', linewidth=0.8)

    norm = BoundaryNorm(np.arange(vmin, vmax, 1), ncolors=cs.WhiteBlueGreenYellowRed.N, clip=True)

    ret = plt.pcolormesh(to_np(lons), to_np(lats), data,
                         transform=crs.PlateCarree(),
                         cmap=cs.WhiteBlueGreenYellowRed,
                         vmin=vmin,
                         norm=norm,
                         vmax=vmax)

    cb = fig.colorbar(ret)
    cb.set_label(label='ppb', fontsize=18, family='Times New Roman')
    cb.set_ticks(np.arange(vmin, vmax, 20))
    cb.ax.tick_params(labelsize=16)
    cb.set_ticklabels(np.arange(vmin, vmax, 20))

    plt.savefig(out, dpi=300)

    plt.close()


if __name__ == '__main__':
    mop_pkl_2017 = r'/home/wrf/lunwen_winter/co_eu_eval/revision/mopitt_data/mopitt_data_2017'
    mop_pkl_2018 = r'/home/wrf/lunwen_winter/co_eu_eval/revision/mopitt_data/mopitt_data_2018'
    mop_pkl_2019 = r'/home/wrf/lunwen_winter/co_eu_eval/revision/mopitt_data/mopitt_data_2019'
    mop_pkl_2020 = r'/home/wrf/lunwen_winter/co_eu_eval/revision/mopitt_data/mopitt_data_2020'

    wrfin = xr.open_dataset('wrfinput_d01')

    with open(mop_pkl_2017, 'rb') as fp:
        frame = pickle.load(fp)

    plot_latlon(frame, wrfin, 'mopitt_xco_2017.png')

    with open(mop_pkl_2018, 'rb') as fp:
        frame = pickle.load(fp)

    plot_latlon(frame, wrfin, 'mopitt_xco_2018.png')

    with open(mop_pkl_2019, 'rb') as fp:
        frame = pickle.load(fp)

    plot_latlon(frame, wrfin, 'mopitt_xco_2019.png')

    with open(mop_pkl_2020, 'rb') as fp:
        frame = pickle.load(fp)

    plot_latlon(frame, wrfin, 'mopitt_xco_2020.png')

    #
    # groups = frame.groupby([frame.year, frame.month])
    #
    # for g in groups:
    #     for i, row in frame.iterrows():
    #         x, y = ll_to_xy(wrfin, row.latitude, row.longitude)
    #
    #         if x >= data.shape[0] or y >= data.shape[1]:
    #             continue
    #
    #         data[x,y] += row.xco
    #         cnt[x,y] +=1
    #
    #     data = data / cnt
    #     data = np.array(data)
    #     print(data.shape)
    #     with open('mopitt_co_data_2017', 'wb') as fp:
    #         pickle.dump(data, fp)
    #
    #
    # with open('mopitt_co_eu_2018', 'rb') as fp:
    #     frame = pickle.load(fp)
    # data = np.zeros(slp.shape)
    # cnt = np.zeros(slp.shape)
    # for i, row in frame.iterrows():
    #     x, y = ll_to_xy(wrfin, row.latitude, row.longitude)
    #
    #     if x >= data.shape[0] or y >= data.shape[1]:
    #         continue
    #
    #     data[x,y] += row.xco
    #     cnt[x,y] +=1
    #
    # data = data / cnt
    # data = np.array(data)
    # print(data.shape)
    # with open('mopitt_co_data_2018', 'wb') as fp:
    #     pickle.dump(data, fp)
    #
    # with open('mopitt_co_eu_2019', 'rb') as fp:
    #     frame = pickle.load(fp)
    # data = np.zeros(slp.shape)
    # cnt = np.zeros(slp.shape)
    # for i, row in frame.iterrows():
    #     x, y = ll_to_xy(wrfin, row.latitude, row.longitude)
    #
    #     if x >= data.shape[0] or y >= data.shape[1]:
    #         continue
    #
    #     data[x,y] += row.xco
    #     cnt[x,y] +=1
    #
    # data = data / cnt
    # data = np.array(data)
    # print(data.shape)
    # with open('mopitt_co_data_2019', 'wb') as fp:
    #     pickle.dump(data, fp)
    #
    # with open('mopitt_co_eu_2020', 'rb') as fp:
    #     frame = pickle.load(fp)
    #
    # data = np.zeros(slp.shape)
    # cnt = np.zeros(slp.shape)
    # for i, row in frame.iterrows():
    #     x, y = ll_to_xy(wrfin, row.latitude, row.longitude)
    #
    #     if x >= data.shape[0] or y >= data.shape[1]:
    #         continue
    #
    #     data[x,y] += row.xco
    #     cnt[x,y] +=1
    #
    # data = data / cnt
    # data = np.array(data)
    # print(data.shape)
    # with open('mopitt_co_data_2020', 'wb') as fp:
    #     pickle.dump(data, fp)

    # lats, lons = latlon_coords(slp)
    #
    # cart_proj = get_cartopy(slp)
    #
    # # Create a figure
    # fig = plt.figure(figsize=(8, 6))
    # # Set the GeoAxes to the projection used by WRF
    # ax = plt.axes(projection=cart_proj)
    #
    # vmin = 60
    # vmax = 85
    #
    #
    # ret = plt.contourf(to_np(lons), to_np(lats), data,
    #                      transform=crs.PlateCarree(),
    #                      cmap=cs.BlGrYeOrReVi200,
    #                      vmin=vmin,
    #                      vmax=vmax,
    #                      levels = 100)
    #
    #
    #
    # cb = fig.colorbar(ret)
    # cb.set_label(label='ppb', fontsize=18, family='Times New Roman')
    # cb.set_ticks(np.arange(vmin, vmax , 1))
    # cb.ax.tick_params(labelsize=16)
    # cb.set_ticklabels(np.arange(vmin, vmax , 1))
    #
    #
    # plt.savefig(r'mopitt_test.png', dpi=300)
    #
    # plt.close()
