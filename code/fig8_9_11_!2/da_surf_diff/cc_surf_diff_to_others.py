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
import glob

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

    fig = plt.figure(figsize=(11, 13))

    ax_2017 = fig.add_subplot(2, 2, 1, projection=cart_proj)
    ax_2018 = fig.add_subplot(2, 2, 2, projection=cart_proj)
    ax_2019 = fig.add_subplot(2, 2, 3, projection=cart_proj)
    ax_2020 = fig.add_subplot(2, 2, 4, projection=cart_proj)

    if exp_type.__contains__('CAM-Chem'):
        fontsize = 13
    else:
        fontsize = 16
    ax_2017.text(0.05, -0.12, '(a) Exp2017_%s' % exp_type, fontsize=fontsize, weight='bold', transform=ax_2017.transAxes)
    ax_2018.text(0.05, -0.12, '(b) Exp2018_%s' % exp_type, fontsize=fontsize, weight='bold', transform=ax_2018.transAxes)
    ax_2019.text(0.05, -0.12, '(c) Exp2019_%s' % exp_type, fontsize=fontsize, weight='bold', transform=ax_2019.transAxes)
    ax_2020.text(0.05, -0.12, '(d) Exp2020_%s' % exp_type, fontsize=fontsize, weight='bold', transform=ax_2020.transAxes)

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
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax_2017.set_xlim(cartopy_xlim(p))
    ax_2017.set_ylim(cartopy_ylim(p))

    gl = ax_2018.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
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
    gl.xlabel_style = {'size': 12}
    gl.ylabel_style = {'size': 12}
    ax_2018.set_xlim(cartopy_xlim(p))
    ax_2018.set_ylim(cartopy_ylim(p))

    gl = ax_2019.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
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

    gl = ax_2020.gridlines(color="black", linestyle="dotted", x_inline=False, y_inline=False, xlocs=[-10, 0, 10, 20],
                           ylocs=[35, 40, 45, 50, 55], draw_labels=False)
    gl.xlabel_style = {'size': 12}
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
    cb.set_label(label='CO (ppb)', fontsize=18, weight='bold')
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
    with open('../da_surf/surf_mean_6h_2017_new_sim_0', 'rb') as fp:
        da_mean_2017 = pickle.load(fp).mean(axis=0)
    with open('../da_surf/surf_mean_6h_2018_new_sim_0', 'rb') as fp:
        da_mean_2018 = pickle.load(fp).mean(axis=0)
    with open('../da_surf/surf_mean_6h_2019_new_sim_0', 'rb') as fp:
        da_mean_2019 = pickle.load(fp).mean(axis=0)
    with open('../da_surf/surf_mean_6h_2020_new_sim_0', 'rb') as fp:
        da_mean_2020 = pickle.load(fp).mean(axis=0)

    with open('../sim_surf/surf_mean_6h_2017_new_sim_0', 'rb') as fp:
        sim_mean_2017 = pickle.load(fp).mean(axis=0)
    with open('../sim_surf/surf_mean_6h_2018_new_sim_0', 'rb') as fp:
        sim_mean_2018 = pickle.load(fp).mean(axis=0)
    with open('../sim_surf/surf_mean_6h_2019_new_sim_0', 'rb') as fp:
        sim_mean_2019 = pickle.load(fp).mean(axis=0)
    with open('../sim_surf/surf_mean_6h_2020_new_sim_0', 'rb') as fp:
        sim_mean_2020 = pickle.load(fp).mean(axis=0)

    with open('mopitt_surf_2017', 'rb') as fp:
        mopitt_mean_2017 = pickle.load(fp)
    with open('mopitt_surf_2018', 'rb') as fp:
        mopitt_mean_2018 = pickle.load(fp)
    with open('mopitt_surf_2019', 'rb') as fp:
        mopitt_mean_2019 = pickle.load(fp)
    with open('mopitt_surf_2020', 'rb') as fp:
        mopitt_mean_2020 = pickle.load(fp)


    with open('cams_surf_2018', 'rb') as fp:
        cams_mean_2018 = pickle.load(fp)
    with open('cams_surf_2019', 'rb') as fp:
        cams_mean_2019 = pickle.load(fp)
    with open('cams_surf_2020', 'rb') as fp:
        cams_mean_2020 = pickle.load(fp)

    with open('ds3137_surf_2017', 'rb') as fp:
        ds3137_mean_2017 = pickle.load(fp)
    with open('ds3137_surf_2018', 'rb') as fp:
        ds3137_mean_2018 = pickle.load(fp)
    with open('ds3137_surf_2019', 'rb') as fp:
        ds3137_mean_2019 = pickle.load(fp)
    with open('ds3137_surf_2020', 'rb') as fp:
        ds3137_mean_2020 = pickle.load(fp)


    # print('=== ds3137 ===')
    # print('ds3137 mean:', np.mean(ds3137_mean_2017),np.mean(ds3137_mean_2018), np.mean(ds3137_mean_2019), np.mean(ds3137_mean_2020))
    # print('diff ds3137 MAE:',
    #       np.mean(np.abs(da_mean_2017 - np.mean(ds3137_mean_2017))),
    #       np.mean(np.abs(da_mean_2018 - np.mean(ds3137_mean_2018))),
    #       np.mean(np.abs(da_mean_2019 - np.mean(ds3137_mean_2019))),
    #       np.mean(np.abs(da_mean_2020 - np.mean(ds3137_mean_2020))))
    #
    # print('diff da_ds3137 STD:',
    #       np.std((da_mean_2017 - np.mean(ds3137_mean_2017))),
    #       np.std((da_mean_2018 - np.mean(ds3137_mean_2018))),
    #       np.std((da_mean_2019 - np.mean(ds3137_mean_2019))),
    #       np.std((da_mean_2020 - np.mean(ds3137_mean_2020))))
    #
    # print('diff sim_ds3137 MAE:',
    #       np.mean(np.abs(sim_mean_2017 - np.mean(ds3137_mean_2017))),
    #       np.mean(np.abs(sim_mean_2018 - np.mean(ds3137_mean_2018))),
    #       np.mean(np.abs(sim_mean_2019 - np.mean(ds3137_mean_2019))),
    #       np.mean(np.abs(sim_mean_2020 - np.mean(ds3137_mean_2020))))
    #
    # print('diff sim_ds3137 STD:',
    #       np.std((sim_mean_2017 - np.mean(ds3137_mean_2017))),
    #       np.std((sim_mean_2018 - np.mean(ds3137_mean_2018))),
    #       np.std((sim_mean_2019 - np.mean(ds3137_mean_2019))),
    #       np.std((sim_mean_2020 - np.mean(ds3137_mean_2020))))
    #
    # print('=== cams ===')
    # print('cams mean:', np.mean(cams_mean_2018), np.mean(cams_mean_2019), np.mean(cams_mean_2020))
    # print('diff da_cams MAE:',
    #       np.mean(np.abs(da_mean_2018 - np.mean(cams_mean_2018, axis=0))),
    #       np.mean(np.abs(da_mean_2019 - np.mean(cams_mean_2019, axis=0))),
    #       np.mean(np.abs(da_mean_2020 - np.mean(cams_mean_2020, axis=0))))
    #
    # print('diff da_cams STD:',
    #       np.std((da_mean_2018 - np.mean(cams_mean_2018, axis=0))),
    #       np.std((da_mean_2019 - np.mean(cams_mean_2019, axis=0))),
    #       np.std((da_mean_2020 - np.mean(cams_mean_2020, axis=0))))
    #
    # print('diff sim_cams MAE:',
    #       np.mean(np.abs(sim_mean_2018 - np.mean(cams_mean_2018, axis=0))),
    #       np.mean(np.abs(sim_mean_2019 - np.mean(cams_mean_2019, axis=0))),
    #       np.mean(np.abs(sim_mean_2020 - np.mean(cams_mean_2020, axis=0))))
    #
    # print('diff sim_cams STD:',
    #       np.std((sim_mean_2018 - np.mean(cams_mean_2018, axis=0))),
    #       np.std((sim_mean_2019 - np.mean(cams_mean_2019, axis=0))),
    #       np.std((sim_mean_2020 - np.mean(cams_mean_2020, axis=0))))
    #
    #
    # print('=== da ===')
    # print('da mean:', np.mean(da_mean_2017), np.mean(da_mean_2018), np.mean(da_mean_2019), np.mean(da_mean_2020))
    #
    # print('mopitt mean:', np.mean(mopitt_mean_2017), np.mean(mopitt_mean_2018), np.mean(mopitt_mean_2019),
    #       np.mean(mopitt_mean_2020))
    #
    # print('mean bias:', np.mean((da_mean_2017 - mopitt_mean_2017)),
    #       np.mean((da_mean_2018 - mopitt_mean_2018)),
    #       np.mean((da_mean_2019 - mopitt_mean_2019)),
    #       np.mean((da_mean_2020 - mopitt_mean_2020)))
    # print('diff MAE:', np.mean(np.abs(da_mean_2017 - mopitt_mean_2017)),
    #       np.mean(np.abs(da_mean_2018 - mopitt_mean_2018)),
    #       np.mean(np.abs(da_mean_2019 - mopitt_mean_2019)),
    #       np.mean(np.abs(da_mean_2020 - mopitt_mean_2020)))
    # print('diff std:', np.std((da_mean_2017 - mopitt_mean_2017)),
    #       np.std((da_mean_2018 - mopitt_mean_2018)),
    #       np.std((da_mean_2019 - mopitt_mean_2019)),
    #       np.std((da_mean_2020 - mopitt_mean_2020)))
    # print('diff RMSE', np.sqrt(np.mean(np.abs(da_mean_2017 - mopitt_mean_2017))),
    #       np.sqrt(np.mean(np.abs(da_mean_2018 - mopitt_mean_2018))),
    #       np.sqrt(np.mean(np.abs(da_mean_2019 - mopitt_mean_2019))),
    #       np.sqrt(np.mean(np.abs(da_mean_2020 - mopitt_mean_2020))))
    # print('=== sim ===')
    # print('sim mean:', np.mean(sim_mean_2017), np.mean(sim_mean_2018), np.mean(sim_mean_2019), np.mean(sim_mean_2020))
    # print('mean bias:', np.mean((sim_mean_2017 - mopitt_mean_2017)),
    #       np.mean((sim_mean_2018 - mopitt_mean_2018)),
    #       np.mean((sim_mean_2019 - mopitt_mean_2019)),
    #       np.mean((sim_mean_2020 - mopitt_mean_2020)))
    # print('diff MAE:', np.mean(np.abs(sim_mean_2017 - mopitt_mean_2017)),
    #       np.mean(np.abs(sim_mean_2018 - mopitt_mean_2018)),
    #       np.mean(np.abs(sim_mean_2019 - mopitt_mean_2019)),
    #       np.mean(np.abs(sim_mean_2020 - mopitt_mean_2020)))
    #
    # print('diff RMSE', np.sqrt(np.mean(np.abs(sim_mean_2017 - mopitt_mean_2017))),
    #       np.sqrt(np.mean(np.abs(sim_mean_2018 - mopitt_mean_2018))),
    #       np.sqrt(np.mean(np.abs(sim_mean_2019 - mopitt_mean_2019))),
    #       np.sqrt(np.mean(np.abs(sim_mean_2020 - mopitt_mean_2020))))
    # add_plot_raw('DA_MOP_SURF', da_mean_2017 - mopitt_mean_2017, da_mean_2018 - mopitt_mean_2018,
    #              da_mean_2019 - mopitt_mean_2019, da_mean_2020 - mopitt_mean_2020, 'wrfinput_d01', '',
    #              './da_mop_surf.png', -200, 200, 40, cs.cmp_b2r)
    #
    # add_plot_raw('DA_CAM-Chem_SURF', da_mean_2017 - ds3137_mean_2017, da_mean_2018 - ds3137_mean_2018,
    #              da_mean_2019 - ds3137_mean_2019, da_mean_2020 - ds3137_mean_2020, 'wrfinput_d01', '',
    #              './da_ds3137_surf.png', -200, 200, 40, cs.cmp_b2r)
    # add_plot_raw('SIM_MOP_SURF', sim_mean_2017 - mopitt_mean_2017, sim_mean_2018 - mopitt_mean_2018,
    #              sim_mean_2019 - mopitt_mean_2019, sim_mean_2020 - mopitt_mean_2020, 'wrfinput_d01', '',
    #              './sim_mop_surf.png', -200, 200, 40, cs.cmp_b2r)
    #
    # add_plot_raw('SIM_CAM-Chem_SURF', sim_mean_2017 - ds3137_mean_2017, sim_mean_2018 - ds3137_mean_2018,
    #              sim_mean_2019 - ds3137_mean_2019, sim_mean_2020 - ds3137_mean_2020, 'wrfinput_d01', '',
    #              './sim_ds3137_surf.png', -200, 200, 40, cs.cmp_b2r)

    # add_plot_raw('MOP_CAM-Chem_SURF', mopitt_mean_2017 - ds3137_mean_2017, mopitt_mean_2018 - ds3137_mean_2018,
    #              mopitt_mean_2019 - ds3137_mean_2019, mopitt_mean_2020 - ds3137_mean_2020, 'wrfinput_d01', '',
    #              './mop_ds3137_surf.png', -200, 200, 40, cs.cmp_b2r)

    # add_plot_raw('SIM',sim_mean_2017, sim_mean_2018, sim_mean_2019, sim_mean_2020 ,os.path.join(inpath_2018, files_2018[0]), '',
    #               '/home/wrf/lunwen_winter/co_eu_eval/cases/final/cc_sim_map/cc_sim_map_new_sim_v2.png', 85, 85, 5, cs.WhiteBlueGreenYellowRed)
