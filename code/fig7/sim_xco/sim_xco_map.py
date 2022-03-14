import matplotlib.pyplot as plt
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from netCDF4 import Dataset, glob
from scipy.ndimage import gaussian_filter1d
from wrf import (to_np, getvar, interplevel, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import numpy as np

import datetime
import os
import matplotlib.ticker as mticker
import cmaps as cs
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

title_dict = {
    'fontsize': 18,
    'weight': 'bold',
    'family': 'Times New Roman'
}
import pickle

def add_plot_raw(exp_type, data_2017, data_2018, data_2019, data_2020, infile, title, outfile, vmin, vmax, vstep, cmap):
    ncfile = Dataset(infile)

    p = getvar(ncfile, 'pressure')[0]

    lats, lons = latlon_coords(p)

    cart_proj = get_cartopy(p)

    fig = plt.figure(figsize=(11,13))

    ax_2017 = fig.add_subplot(2, 2, 1, projection=cart_proj)
    ax_2018 = fig.add_subplot(2, 2, 2, projection=cart_proj)
    ax_2019 = fig.add_subplot(2, 2, 3, projection=cart_proj)
    ax_2020 = fig.add_subplot(2, 2, 4, projection=cart_proj)

    ax_2017.text(0.2, -0.12,'(a) Exp2017_%s' % exp_type, fontsize=16, weight='bold', transform=ax_2017.transAxes)
    ax_2018.text(0.2,-0.12, '(b) Exp2018_%s' % exp_type,fontsize=16, weight='bold', transform=ax_2018.transAxes)
    ax_2019.text(0.2,-0.12, '(c) Exp2019_%s' % exp_type, fontsize=16, weight='bold', transform=ax_2019.transAxes)
    ax_2020.text(0.2,-0.12, '(d) Exp2020_%s'  % exp_type, fontsize=16, weight='bold', transform=ax_2020.transAxes)



    # ax = plt.axes(projection=cart_proj)

    # Download and add the states and coastlines
    states = NaturalEarthFeature(category="cultural", scale="50m",
                                 facecolor="none",
                                 name="admin_1_states_provinces_shp")
    ax_2017.add_feature(states, linewidth=.5, edgecolor="black")
    ax_2017.coastlines('50m', linewidth=0.8)
    ax_2018.add_feature(states, linewidth=.5, edgecolor="black")
    ax_2018.coastlines('50m', linewidth=0.8)
    ax_2019.add_feature(states, linewidth=.5, edgecolor="black")
    ax_2019.coastlines('50m', linewidth=0.8)
    ax_2020.add_feature(states, linewidth=.5, edgecolor="black")
    ax_2020.coastlines('50m', linewidth=0.8)

    data_2017[data_2017 > vmax] = vmax
    data_2017[data_2017 < vmin] = vmin
    data_2018[data_2018 > vmax] = vmax
    data_2018[data_2018 < vmin] = vmin
    data_2019[data_2019 > vmax] = vmax
    data_2019[data_2019 < vmin] = vmin
    data_2020[data_2020 > vmax] = vmax
    data_2020[data_2020 < vmin] = vmin
    ret = ax_2017.pcolormesh(to_np(lons), to_np(lats), data_2017,
                       transform=crs.PlateCarree(),
                       cmap=cmap,
                       # levels=np.arange(vmin, vmax + 1, 1),
                       vmax=vmax,
                       vmin=vmin)
    ret = ax_2018.pcolormesh(to_np(lons), to_np(lats), data_2018,
                       transform=crs.PlateCarree(),
                       cmap=cmap,
                       # levels=np.arange(vmin, vmax + 1, 1),
                       vmax=vmax,
                       vmin=vmin)
    ret = ax_2019.pcolormesh(to_np(lons), to_np(lats), data_2019,
                       transform=crs.PlateCarree(),
                       cmap=cmap,
                       # levels=np.arange(vmin, vmax + 1, 1),
                       vmax=vmax,
                       vmin=vmin)
    ret = ax_2020.pcolormesh(to_np(lons), to_np(lats), data_2020,
                       transform=crs.PlateCarree(),
                       cmap=cmap,
                       # levels=np.arange(vmin, vmax + 1, 1),
                       vmax=vmax,
                       vmin=vmin)
    # Set the map bounds

    gl = ax_2017.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
                           ylocs=[35, 40, 45, 50,55],draw_labels=False)

    # gl.top_labels=False
    # gl.right_labels=False
    # gl.bottom_labels = True
    # gl.x_inline=False
    # gl.y_inline=False
    # gl.xlocator=mticker.FixedLocator([-10.0, 0.0, 10.0, 20.0])
    # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
    # gl.xformatter=LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size':12}
    gl.ylabel_style = {'size': 12}
    ax_2017.set_xlim(cartopy_xlim(p))
    ax_2017.set_ylim(cartopy_ylim(p))

    gl = ax_2018.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
                           ylocs=[35, 40, 45, 50,55],draw_labels=False)

    # gl.top_labels=False
    # gl.right_labels=False
    # gl.bottom_labels = True
    # gl.x_inline=False
    # gl.y_inline=False
    # gl.xlocator=mticker.FixedLocator([-10.0, 0.0, 10.0, 20.0])
    # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
    # gl.xformatter=LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style={'size':12}
    gl.ylabel_style = {'size': 12}
    ax_2018.set_xlim(cartopy_xlim(p))
    ax_2018.set_ylim(cartopy_ylim(p))

    gl = ax_2019.gridlines(color="black", linestyle="dotted",x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
                           ylocs=[35, 40, 45, 50,55],draw_labels=False)

    gl.xlabel_style={'size':12}
    gl.ylabel_style = {'size': 12}
    ax_2019.set_xlim(cartopy_xlim(p))
    ax_2019.set_ylim(cartopy_ylim(p))
    # gl.top_labels=False
    # gl.right_labels=False
    # gl.bottom_labels = True
    # gl.x_inline=False
    # gl.y_inline=False
    # gl.xlocator=mticker.FixedLocator([-10, 0, 10, 20])
    # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
    # gl.xformatter=LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style={'size':16}
    # gl.ylabel_style = {'size': 16}


    gl = ax_2020.gridlines(color="black", linestyle="dotted",x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
                           ylocs=[35, 40, 45, 50,55],draw_labels=False)
    gl.xlabel_style={'size':12}
    gl.ylabel_style = {'size': 12}
    ax_2020.set_xlim(cartopy_xlim(p))
    ax_2020.set_ylim(cartopy_ylim(p))

    # plt.subplots_adjust(left=0.05, right=0.95)
    # gl.top_labels=False
    # gl.right_labels=False
    # gl.bottom_labels = True
    # gl.xformatter=LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    # gl.xlabel_style={'size':16}
    # gl.ylabel_style = {'size': 16}

    # plt.axes(0.1,0,1,0.)
    # plt.subplots_adjust(left=0.05, right=0.95)
    cb = plt.colorbar(ret, ax=[ax_2017, ax_2018, ax_2019, ax_2020], orientation='horizontal',shrink=0.8, pad=0.1, aspect=30, fraction=0.1, spacing='proportional')
    cb.set_label(label='XCO (ppb)', fontsize=18, weight='bold')
    cb.set_ticks(np.arange(vmin, vmax + 1, vstep))
    cb.ax.tick_params(labelsize=16)
    cb.set_ticklabels(np.arange(vmin, vmax + 1, vstep))

    # print(cb)
    # plt.clim(vmin, vmax)

    # plt.clim(vmin, vmax)
    # plt.colorbar(ret, ax=ax, label)

    # plt.quiver(to_np(lons[::5, ::5]), to_np(lats[::5, ::5]),
    #            to_np(u_950[::5, ::5]), to_np(v_950[::5, ::5]),
    #            transform=crs.PlateCarree())



    # Add the gridlines

    # gl.bottom_labels=False
    # if text != '':

    #     plt.annotate(text, (0.11, 0.88), xycoords='figure fraction', fontsize=18, family='Times New Roman',
    #                  color='black', weight='bold', backgroundcolor='white')
    # if text != '':
    #     plt.title(title + '(%s)' % text, fontsize=18)
    # else:
    # plt.title(title, fontsize=18)

    plt.savefig(outfile, dpi=300)
    plt.close()

if __name__ == '__main__':
    # with open('da_pixel_mean_2017', 'rb') as fp:
    #     da_mean_2017 = pickle.load(fp)
    # with open('da_pixel_mean_2018', 'rb') as fp:
    #     da_mean_2018 = pickle.load(fp)
    # with open('da_pixel_mean_2019', 'rb') as fp:
    #     da_mean_2019 = pickle.load(fp)
    # with open('da_pixel_mean_2020', 'rb') as fp:
    #     da_mean_2020 = pickle.load(fp)
    with open('sim_xco_2017', 'rb') as fp:
        sim_mean_2017 = pickle.load(fp).mean(axis=0)
    with open('sim_xco_2018', 'rb') as fp:
        sim_mean_2018 = pickle.load(fp).mean(axis=0)
    with open('sim_xco_2019', 'rb') as fp:
        sim_mean_2019 = pickle.load(fp).mean(axis=0)
    with open('sim_xco_2020', 'rb') as fp:
        sim_mean_2020 = pickle.load(fp).mean(axis=0)

    # with open('sim_pixel_mean_2017', 'rb') as fp:
    #     sim_mean_2017 = pickle.load(fp)
    # with open('sim_pixel_mean_2018', 'rb') as fp:
    #     sim_mean_2018 = pickle.load(fp)
    # with open('sim_pixel_mean_2019', 'rb') as fp:
    #     sim_mean_2019 = pickle.load(fp)
    # with open('sim_pixel_mean_2020', 'rb') as fp:
    #     sim_mean_2020 = pickle.load(fp)




    print(2017, 'sim:', np.mean(sim_mean_2017))
    print(2018, 'sim:', np.mean(sim_mean_2018))
    print(2019, 'sim:',np.mean(sim_mean_2019))
    print(2020, 'sim:',np.mean(sim_mean_2020))


    add_plot_raw('SIM',sim_mean_2017, sim_mean_2018, sim_mean_2019, sim_mean_2020 ,'../wrfinput_d01', '',
                  'sim_xco_map.png', 60, 85, 5, cs.WhiteBlueGreenYellowRed)


