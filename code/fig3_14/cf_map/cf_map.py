import matplotlib.pyplot as plt
from cartopy import crs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cartopy.mpl.ticker import LatitudeLocator, LongitudeFormatter, LatitudeFormatter, LongitudeLocator
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, ListedColormap
from netCDF4 import Dataset, glob
from scipy.ndimage import gaussian_filter1d
from wrf import (to_np, getvar, interplevel, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import numpy as np

import datetime
import os
import matplotlib.ticker as mticker
from pandas.plotting import register_matplotlib_converters
import cmaps as cs

register_matplotlib_converters()

title_dict = {
    'fontsize': 18,
    'weight': 'bold',
    'family': 'Times New Roman'
}
import pickle


# def add_plot_raw(data_2018, data_2019, data_2020, infile, title, outfile, vmin, vmax, vstep, cmap=get_cmap('jet')):
#     ncfile = Dataset(infile)
#
#     p = getvar(ncfile, 'pressure')[0]
#
#     lats, lons = latlon_coords(p)
#
#     cart_proj = get_cartopy(p)
#
#     fig = plt.figure(figsize=(20, 8))
#
#     ax_2018 = fig.add_subplot(1, 3, 1, projection=cart_proj)
#     ax_2019 = fig.add_subplot(1, 3, 2, projection=cart_proj)
#     ax_2020 = fig.add_subplot(1, 3, 3, projection=cart_proj)
#     ax_2018.text(0.43, -0.1, '2018', fontsize=18, weight='bold', transform=ax_2018.transAxes)
#     ax_2019.text(0.43, -0.1, '2019', fontsize=18, weight='bold', transform=ax_2019.transAxes)
#     ax_2020.text(0.43, -0.1, '2020', fontsize=18, weight='bold', transform=ax_2020.transAxes)
#
#     # ax = plt.axes(projection=cart_proj)
#
#     # Download and add the states and coastlines
#     states = NaturalEarthFeature(category="cultural", scale="50m",
#                                  facecolor="none",
#                                  name="admin_1_states_provinces_shp")
#     ax_2018.add_feature(states, linewidth=.5, edgecolor="black")
#     ax_2018.coastlines('50m', linewidth=0.8)
#     ax_2019.add_feature(states, linewidth=.5, edgecolor="black")
#     ax_2019.coastlines('50m', linewidth=0.8)
#     ax_2020.add_feature(states, linewidth=.5, edgecolor="black")
#     ax_2020.coastlines('50m', linewidth=0.8)
#
#     data_2018[data_2018 > vmax] = vmax
#     data_2018[data_2018 < vmin] = vmin
#     data_2019[data_2019 > vmax] = vmax
#     data_2019[data_2019 < vmin] = vmin
#     data_2020[data_2020 > vmax] = vmax
#     data_2020[data_2020 < vmin] = vmin
#     ret = ax_2018.contourf(to_np(lons), to_np(lats), data_2018,
#                            transform=crs.PlateCarree(),
#                            cmap=cmap,
#                            levels=np.arange(vmin, vmax + 1, 1),
#                            vmax=vmax,
#                            vmin=vmin)
#     ret = ax_2019.contourf(to_np(lons), to_np(lats), data_2019,
#                            transform=crs.PlateCarree(),
#                            cmap=cmap,
#                            levels=np.arange(vmin, vmax + 1, 1),
#                            vmax=vmax,
#                            vmin=vmin)
#     ret = ax_2020.contourf(to_np(lons), to_np(lats), data_2020,
#                            transform=crs.PlateCarree(),
#                            cmap=cmap,
#                            levels=np.arange(vmin, vmax + 1, 1),
#                            vmax=vmax,
#                            vmin=vmin)
#     # Set the map bounds
#
#     gl = ax_2018.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
#                            xlocs=[-10, 0, 10, 20],
#                            ylocs=[35, 40, 45, 50, 55], draw_labels=True)
#
#     # gl.top_labels=False
#     # gl.right_labels=False
#     # gl.bottom_labels = True
#     # gl.x_inline=False
#     # gl.y_inline=False
#     # gl.xlocator=mticker.FixedLocator([-10.0, 0.0, 10.0, 20.0])
#     # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
#     # gl.xformatter=LONGITUDE_FORMATTER
#     # gl.yformatter = LATITUDE_FORMATTER
#     gl.xlabel_style = {'size': 12}
#     gl.ylabel_style = {'size': 12}
#     ax_2018.set_xlim(cartopy_xlim(p))
#     ax_2018.set_ylim(cartopy_ylim(p))
#
#     gl = ax_2019.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
#                            xlocs=[-10, 0, 10, 20],
#                            ylocs=[35, 40, 45, 50, 55], draw_labels=True)
#
#     gl.xlabel_style = {'size': 12}
#     gl.ylabel_style = {'size': 12}
#     ax_2019.set_xlim(cartopy_xlim(p))
#     ax_2019.set_ylim(cartopy_ylim(p))
#     # gl.top_labels=False
#     # gl.right_labels=False
#     # gl.bottom_labels = True
#     # gl.x_inline=False
#     # gl.y_inline=False
#     # gl.xlocator=mticker.FixedLocator([-10, 0, 10, 20])
#     # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
#     # gl.xformatter=LONGITUDE_FORMATTER
#     # gl.yformatter = LATITUDE_FORMATTER
#     # gl.xlabel_style={'size':16}
#     # gl.ylabel_style = {'size': 16}
#
#     gl = ax_2020.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
#                            xlocs=[-10, 0, 10, 20],
#                            ylocs=[35, 40, 45, 50, 55], draw_labels=True)
#     gl.xlabel_style = {'size': 12}
#     gl.ylabel_style = {'size': 12}
#     ax_2020.set_xlim(cartopy_xlim(p))
#     ax_2020.set_ylim(cartopy_ylim(p))
#
#     plt.subplots_adjust(left=0.05, right=0.95)
#
#     cb = plt.colorbar(ret, ax=[ax_2018, ax_2019, ax_2020], orientation='horizontal', shrink=0.8, pad=0.15,
#                       aspect=60, fraction=0.1)
#
#     cb.set_label(label='ppb', fontsize=18, family='Times New Roman')
#     cb.set_ticks(np.arange(vmin, vmax + 1, vstep))
#     cb.ax.tick_params(labelsize=16)
#     cb.set_ticklabels(np.arange(vmin, vmax + 1, vstep))
#
#     plt.savefig(outfile, dpi=300)
#     plt.close()


def add_plot_raw(type, data_2017, data_2018, data_2019, data_2020, infile, title, outfile, vmin, vmax, levels, cmap=get_cmap('jet')):
    ncfile = Dataset(infile)

    p = getvar(ncfile, 'pressure')[0]

    lats, lons = latlon_coords(p)

    cart_proj = get_cartopy(p)

    fig = plt.figure(figsize=(11, 13))

    ax_2017 = fig.add_subplot(2, 2, 1, projection=cart_proj)
    ax_2018 = fig.add_subplot(2, 2, 2, projection=cart_proj)
    ax_2019 = fig.add_subplot(2, 2, 3, projection=cart_proj)
    ax_2020 = fig.add_subplot(2, 2, 4, projection=cart_proj)
    ax_2017.text(0.2, -0.12, '(a) Exp2017_%s' % type, fontsize=16, weight='bold', transform=ax_2017.transAxes)
    ax_2018.text(0.2, -0.12, '(b) Exp2018_%s' % type, fontsize=16, weight='bold', transform=ax_2018.transAxes)
    ax_2019.text(0.2, -0.12, '(c) Exp2019_%s' % type, fontsize=16, weight='bold', transform=ax_2019.transAxes)
    ax_2020.text(0.2, -0.12, '(d) Exp2020_%s' % type, fontsize=16, weight='bold', transform=ax_2020.transAxes)

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

    ret = ax_2017.contourf(to_np(lons), to_np(lats), data_2017,
                           transform=crs.PlateCarree(),
                           colors=cmap,
                           # cmap=cmap,
                           levels=levels,
                           vmax=vmax,
                           vmin=vmin)
    ret = ax_2018.contourf(to_np(lons), to_np(lats), data_2018,
                           transform=crs.PlateCarree(),
                           colors=cmap,
                           # cmap=cmap,
                           levels=levels,
                           vmax=vmax,
                           vmin=vmin)
    ret = ax_2019.contourf(to_np(lons), to_np(lats), data_2019,
                           transform=crs.PlateCarree(),
                           colors=cmap,
                           # cmap=cmap,
                           levels=levels,
                           vmax=vmax,
                           vmin=vmin)
    ret = ax_2020.contourf(to_np(lons), to_np(lats), data_2020,
                           transform=crs.PlateCarree(),
                           colors=cmap,
                           # cmap=cmap,
                           levels=levels,
                           vmax=vmax,
                           vmin=vmin)
    # Set the map bounds

    gl = ax_2017.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
                           xlocs=[-10, 0, 10, 20],
                           xformatter=LongitudeFormatter(),
                           yformatter=LatitudeFormatter(),
                           ylocs=[35, 40, 45, 50, 55], draw_labels=False)

    gl.xlabel_style = {'size': 12, 'style': 'normal'}
    gl.ylabel_style = {'size': 12, 'style': 'normal'}
    ax_2017.set_xlim(cartopy_xlim(p))
    ax_2017.set_ylim(cartopy_ylim(p))

    gl = ax_2018.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
                           xlocs=[-10, 0, 10, 20],
                           xformatter=LongitudeFormatter(),
                           yformatter=LatitudeFormatter(),
                           ylocs=[35, 40, 45, 50, 55], draw_labels=False)


    # gl.top_labels=False
    # gl.right_labels=False
    # gl.bottom_labels = True
    # gl.x_inline=False
    # gl.y_inline=False
    # gl.xlocator=mticker.FixedLocator([-10.0, 0.0, 10.0, 20.0])
    # gl.ylocator = mticker.FixedLocator([35, 40, 45, 50,55])
    # gl.xformatter=LONGITUDE_FORMATTER
    # gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'style': 'normal'}
    gl.ylabel_style = {'size': 12, 'style': 'normal'}
    ax_2018.set_xlim(cartopy_xlim(p))
    ax_2018.set_ylim(cartopy_ylim(p))

    gl = ax_2019.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
                           xlocs=[-10, 0, 10, 20],
                           xformatter=LongitudeFormatter(),
                           yformatter=LatitudeFormatter(),
                           ylocs=[35, 40, 45, 50, 55], draw_labels=False)

    gl.xlabel_style = {'size': 12}
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

    gl = ax_2020.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False,
                           xlocs=[-10, 0, 10, 20],
                           xformatter=LongitudeFormatter(auto_hide=False),
                           yformatter=LatitudeFormatter(auto_hide=False),
                           ylocs=[35, 40, 45, 50, 55], draw_labels=False)
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax_2020.set_xlim(cartopy_xlim(p))
    ax_2020.set_ylim(cartopy_ylim(p))

    #plt.subplots_adjust(left=0.05, right=0.95)

    if type == 'DIFF':
        cb = plt.colorbar(ret, ax=[ax_2017, ax_2018, ax_2019, ax_2020], orientation='horizontal', shrink=0.95, pad=0.1,
                          aspect=30, fraction=0.1)
        cb.ax.tick_params(labelsize=10)
    else:
        cb = plt.colorbar(ret, ax=[ax_2017, ax_2018, ax_2019, ax_2020], orientation='horizontal', shrink=0.8, pad=0.1,
                      aspect=30, fraction=0.1)
        cb.ax.tick_params(labelsize=16)

    cb.set_label(label='Flux (mol $\mathregular{km^{-2}}$ $\mathregular{h^{-1}}$)', fontsize=18, weight='bold')
    cb.set_ticks(levels)

    cb.set_ticklabels(levels)

    plt.savefig(outfile, dpi=300)
    plt.close()
if __name__ == '__main__':

    with open('sum_prior_2017', 'rb') as fp:
        sum_prior_2017 = pickle.load(fp)
    with open('sum_prior_2018', 'rb') as fp:
        sum_prior_2018 = pickle.load(fp)
    with open('sum_prior_2019', 'rb') as fp:
        sum_prior_2019 = pickle.load(fp)
    with open('sum_prior_2020', 'rb') as fp:
        sum_prior_2020 = pickle.load(fp)
    with open('sum_prior_2017_wo', 'rb') as fp:
        sum_prior_2017_wo = pickle.load(fp)
    with open('sum_prior_2018_wo', 'rb') as fp:
        sum_prior_2018_wo = pickle.load(fp)
    with open('sum_prior_2019_wo', 'rb') as fp:
        sum_prior_2019_wo = pickle.load(fp)
    with open('sum_prior_2020_wo', 'rb') as fp:
        sum_prior_2020_wo = pickle.load(fp)

    with open('sum_post_2017', 'rb') as fp:
        sum_post_2017 = pickle.load(fp)
    with open('sum_post_2018', 'rb') as fp:
        sum_post_2018 = pickle.load(fp)
    with open('sum_post_2019', 'rb') as fp:
        sum_post_2019 = pickle.load(fp)
    with open('sum_post_2020', 'rb') as fp:
        sum_post_2020 = pickle.load(fp)
    with open('sum_post_2017_wo', 'rb') as fp:
        sum_post_2017_wo = pickle.load(fp)
    with open('sum_post_2018_wo', 'rb') as fp:
        sum_post_2018_wo = pickle.load(fp)
    with open('sum_post_2019_wo', 'rb') as fp:
        sum_post_2019_wo = pickle.load(fp)
    with open('sum_post_2020_wo', 'rb') as fp:
        sum_post_2020_wo = pickle.load(fp)

    # 打开文件
    inpath_2020 = r"/home/wrf/lunwen_winter/data/CC/2020/ALL"
    title_2020 = 'assimilated XCO mean in winter 2020'
    inpath_2019 = r"/home/wrf/lunwen_winter/data/CC/2019/ALL"
    title_2019 = 'assimilated XCO mean in winter 2019'
    inpath_2018 = r"/home/wrf/lunwen_winter/data/CC/2018/ALL"
    title_2018 = 'assimilated XCO mean in winter 2018'
    files_2018 = os.listdir(inpath_2018)
    files_2019 = os.listdir(inpath_2019)
    files_2020 = os.listdir(inpath_2020)

    print(2017, 'da:', np.mean(sum_post_2017))
    print(2018, 'da:', np.mean(sum_post_2018))
    print(2019, 'da:', np.mean(sum_post_2019))
    print(2020, 'da:', np.mean(sum_post_2020))
    print(2017, 'sim:', np.mean(sum_prior_2017))
    print(2018, 'sim:', np.mean(sum_prior_2018))
    print(2019, 'sim:', np.mean(sum_prior_2019))
    print(2020, 'sim:', np.mean(sum_prior_2020))
    print(2017, 'diff:', np.mean(sum_post_2017) -np.mean(sum_prior_2017))
    print(2018, 'diff:', np.mean(sum_post_2018) - np.mean(sum_prior_2018))
    print(2019, 'diff:', np.mean(sum_post_2019) - np.mean(sum_prior_2019))
    print(2020, 'diff:', np.mean(sum_post_2020) - np.mean(sum_prior_2020))


    # cus_cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
    add_plot_raw('DA', np.nanmean(sum_post_2017, axis=0), np.nanmean(sum_post_2018, axis=0), np.nanmean(sum_post_2019, axis=0), np.nanmean(sum_post_2020, axis=0), os.path.join(inpath_2018, files_2018[0]), '',
                 '/home/wrf/lunwen_winter/co_eu_eval/cases/final/cf_map/cf_post_map_v2.png', 0, 1000, [0, 0.05, 0.1, 0.5, 3, 5, 15, 25, 50, 100, 1000], cs.perc2_9lev.colors)
    add_plot_raw('PRIOR', np.nanmean(sum_prior_2017, axis=0), np.nanmean(sum_prior_2018, axis=0),
                 np.nanmean(sum_prior_2019, axis=0), np.nanmean(sum_prior_2020, axis=0),
                 os.path.join(inpath_2018, files_2018[0]), '',
                 '/home/wrf/lunwen_winter/co_eu_eval/cases/final/cf_map/cf_prior_map_v2.png', 0, 1000,
                 [0, 0.05, 0.1, 0.5, 3, 5, 15, 25, 50, 100, 1000], cs.perc2_9lev.colors)
    add_plot_raw('DIFF', np.nanmean(sum_post_2017, axis=0) - np.nanmean(sum_prior_2017, axis=0),
                 np.nanmean(sum_post_2018, axis=0) - np.nanmean(sum_prior_2018, axis=0),
                 np.nanmean(sum_post_2019, axis=0) - np.nanmean(sum_prior_2019, axis=0),
                 np.nanmean(sum_post_2020, axis=0) - np.nanmean(sum_prior_2020, axis=0), os.path.join(inpath_2018, files_2018[0]), '',
                 '/home/wrf/lunwen_winter/co_eu_eval/cases/final/cf_map/cf_diff_map_v2.png', -1000, 1000, [-1000, -100, -50, -25, -15, -5, -3, -0.5, -0.1,-0.05, 0, 0.05, 0.1, 0.5, 3, 5, 15, 25, 50, 100, 1000], cs.cmp_flux.colors)
    # add_plot_raw(np.nanmean(sum_post_2018, axis=0), np.nanmean(sum_post_2019, axis=0), np.nanmean(sum_post_2020, axis=0), os.path.join(inpath_2018, files_2018[0]), '',
    #              '/home/wrf/lunwen_winter/co_eu_eval/cases/final/cf_map/cf_map.png', 0, 200,20, cs.WhiteYellowOrangeRed)




