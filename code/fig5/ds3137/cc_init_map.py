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
import xesmf as xe
register_matplotlib_converters()

title_dict = {
    'fontsize': 18,
    'weight': 'bold',
    'family': 'Times New Roman'
}
import pickle

def add_plot_raw(data_2017, data_2018, data_2019, data_2020,lat_in, lon_in, infile, title, outfile, vmin, vmax, vstep, cmap):



    ncfile = Dataset(infile)

    p = getvar(ncfile, 'pressure')[0]

    lats, lons = latlon_coords(p)

    lon_tmp = []
    lat_tmp = []
    mesh_lon_in, mesh_lat_in = np.meshgrid(lon_in, lat_in)
    data_2017_new = []
    data_2018_new = []
    data_2019_new = []
    data_2020_new = []
    mesh_lon_in_new = []
    mesh_lat_in_new = []
    for c in range(21,len(lon_in)):
        data_2017_new.append(data_2017[:, c])
        data_2018_new.append(data_2018[:, c])
        data_2019_new.append(data_2019[:, c])
        data_2020_new.append(data_2020[:, c])
        mesh_lon_in_new.append(mesh_lon_in[:,c].transpose() -360)
        mesh_lat_in_new.append(mesh_lat_in[:,c].transpose())

    for c in range(0,21):
        data_2017_new.append(data_2017[:, c])
        data_2018_new.append(data_2018[:, c])
        data_2019_new.append(data_2019[:, c])
        data_2020_new.append(data_2020[:, c])
        mesh_lon_in_new.append(mesh_lon_in[:,c].transpose())
        mesh_lat_in_new.append(mesh_lat_in[:,c].transpose())



    mesh_lon_out = lons
    mesh_lat_out = lats

    grid_in = {'lon': np.array(mesh_lon_in_new).transpose(), 'lat': np.array(mesh_lat_in_new).transpose()}
    grid_out = {'lon': mesh_lon_out.data, 'lat': mesh_lat_out.data}
    regridder = xe.Regridder(grid_in, grid_out, 'bilinear')
    data_2017 = regridder(np.array(data_2017_new).transpose())
    data_2018 = regridder(np.array(data_2018_new).transpose())
    data_2019 = regridder(np.array(data_2019_new).transpose())
    data_2020 = regridder(np.array(data_2020_new).transpose())

    with open('ds3137_xco_2017', 'wb') as fp:
        pickle.dump(data_2017, fp)

    with open('ds3137_xco_2018', 'wb') as fp:
        pickle.dump(data_2018, fp)

    with open('ds3137_xco_2019', 'wb') as fp:
        pickle.dump(data_2019, fp)

    with open('ds3137_xco_2020', 'wb') as fp:
        pickle.dump(data_2020, fp)

    print(2017, 'init', np.mean(data_2017))
    print(2018, 'init:', np.mean(data_2018))
    print(2019, 'init:', np.mean(data_2019))
    print(2020, 'init:', np.mean(data_2020))

    cart_proj = get_cartopy(p)

    fig = plt.figure(figsize=(11, 13))

    ax_2017 = fig.add_subplot(2, 2, 1, projection=cart_proj)
    ax_2018 = fig.add_subplot(2, 2, 2, projection=cart_proj)
    ax_2019 = fig.add_subplot(2, 2, 3, projection=cart_proj)
    ax_2020 = fig.add_subplot(2, 2, 4, projection=cart_proj)
    ax_2017.text(0.1, -0.12, '(a) Exp2017_CAM-Chem', fontsize=16, weight='bold', transform=ax_2017.transAxes)
    ax_2018.text(0.1, -0.12, '(b) Exp2018_CAM-Chem',fontsize=16, weight='bold', transform=ax_2018.transAxes)
    ax_2019.text(0.1, -0.12, '(c) Exp2019_CAM-Chem', fontsize=16, weight='bold', transform=ax_2019.transAxes)
    ax_2020.text(0.1, -0.12, '(d) Exp2020_CAM-Chem', fontsize=16, weight='bold', transform=ax_2020.transAxes)



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
    cb = plt.colorbar(ret, ax=[ax_2017, ax_2018, ax_2019, ax_2020], orientation='horizontal', shrink=0.8, pad=0.1,
                      aspect=30, fraction=0.1)
    cb.set_label(label='XCO (ppb)', fontsize=18, weight='bold', family='Times New Roman')
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
    with open('ds3137_co_2017_winter', 'rb') as fp:
        init_2017 = pickle.load(fp)
    with open('ds3137_co_2018_winter', 'rb') as fp:
        init_2018 = pickle.load(fp)
    with open('ds3137_co_2019_winter', 'rb') as fp:
        init_2019 = pickle.load(fp)
    with open('ds3137_co_2020_winter', 'rb') as fp:
        init_2020 = pickle.load(fp)
    with open('ds3137_data_lat', 'rb') as fp:
        init_lat = pickle.load(fp)
    with open('ds3137_data_lon', 'rb') as fp:
        init_lon = pickle.load(fp)


    # 打开文件
    inpath_2020 = r"/home/wrf/lunwen_winter/data/CC/2020/ALL"
    title_2020 = 'XCO mean in winter 2020(DA)'
    inpath_2019 = r"/home/wrf/lunwen_winter/data/CC/2019/ALL"
    title_2019 = 'XCO mean in winter 2019(DA)'
    inpath_2018 = r"/home/wrf/lunwen_winter/data/CC/2018/ALL"
    title_2018 = 'XCO mean in winter 2018(DA)'
    files_2018 = os.listdir(inpath_2018)
    files_2019 = os.listdir(inpath_2019)
    files_2020 = os.listdir(inpath_2020)




    add_plot_raw(init_2017.mean(axis=0),init_2018.mean(axis=0), init_2019.mean(axis=0), init_2020.mean(axis=0), init_lat, init_lon,os.path.join(inpath_2018, files_2018[0]), '',
                  '/home/wrf/lunwen_winter/co_eu_eval/cases/final/ds3137/cc_init_map_v2.png', 60, 85, 5, cs.WhiteBlueGreenYellowRed)





