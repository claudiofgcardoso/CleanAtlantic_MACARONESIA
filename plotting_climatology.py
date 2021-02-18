#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:23:52 2019

@author: Ubuntu
"""
import socket 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RectBivariateSpline
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import netCDF4
#import xarray as xr
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from matplotlib.patches import Polygon

#%% Loading data
class ncData(object):
    def __init__(self):
        self.lon = []
        self.lat = []
        self.U = []
        self.V = []

def loading_data(path, start_date, finish_date):
    Tall=ncData()
    
    fdir=sorted(glob(path + '*.nc'))
    len_path=len(path)   

    start_date_str=start_date.strftime("%Y%m%d")
    finish_date_str=finish_date.strftime("%Y%m%d")
    for i, e in enumerate(fdir):
        if e.find(str(start_date_str),len_path) > 0:
            start_index = i 
        elif e.find(str(finish_date_str),len_path) > 0:
            finish_index = i 
                
    filenames = fdir[start_index:finish_index+1]  

    if path.find('0083') > 0:
        variables = {'U': 'uo',
                     'V': 'vo',
                     'lon': 'longitude',
                     'lat': 'latitude'}
    elif path.find('025') > 0:
        variables = {'U': 'u',
                     'V': 'v',
                     'lon': 'longitude',
                     'lat': 'latitude'}   
    elif path.find('WW3') > 0:
        variables = {'U': 'uuss',
                     'V': 'vuss',
                     'lon': 'longitude',
                     'lat': 'latitude'}
    elif path.find('WIND') > 0:
        variables = {'U': 'uwnd',
                     'V': 'vwnd',
                     'lon': 'longitude',
                     'lat': 'latitude'}
            
#    try:
    # nc = netCDF4.MFDataset(filenames, aggdim='time')
#    if path.find('0083') > 0 or path.find('025') > 0:
#        temp_u=nc.variables[variables['U']][:,0,:].data
#        temp_v=nc.variables[variables['V']][:,0,:].data
#    elif path.find('WW3') > 0 or path.find('WIND') > 0:
#        temp_u=nc.variables[variables['U']][:].data
#        temp_v=nc.variables[variables['V']][:].data 
#    nc.close()
#    except:
    print('MFDataset did not work. Doing the go around')
    for fn, fi in enumerate(filenames):
        print('Reading %s' %fi)
        nc = netCDF4.Dataset(fi)
        if fn == 0:
            nc = netCDF4.Dataset(filenames[0])
            Tall.lon=nc.variables[variables['lon']][:]
            Tall.lat=nc.variables[variables['lat']][:]
    
            if path.find('0083') > 0 or path.find('025') > 0:
                ex_u=nc.variables[variables['U']][:,0,:].data.shape
                ex_v=nc.variables[variables['V']][:,0,:].data.shape  
            elif path.find('WW3') > 0 or path.find('WIND') > 0:
                ex_u=nc.variables[variables['U']][:].data.shape
                ex_v=nc.variables[variables['V']][:].data.shape
                
            fill_value=nc.variables[variables['U']]._FillValue
            temp_dim=len(filenames)
            temp_u=np.empty([temp_dim, ex_u[1], ex_u[2]])
            temp_v=np.empty([temp_dim, ex_v[1], ex_v[2]])
            temp_u[:]=np.nan
            temp_v[:]=np.nan
        
        if path.find('0083') > 0 or path.find('025') > 0:
            temp_u[fn,:]=nc.variables[variables['U']][:,0,:].data
            temp_v[fn,:]=nc.variables[variables['V']][:,0,:].data
        elif path.find('WW3') > 0 or path.find('WIND') > 0:
            temp_u[fn,:]=nc.variables[variables['U']][:].data
            temp_v[fn,:]=nc.variables[variables['V']][:].data 
        
        nc.close()
        
    temp_u[temp_u==fill_value]=np.nan
    temp_v[temp_v==fill_value]=np.nan
    temp_u[temp_u>300]=np.nan
    temp_u[temp_u<-300]=np.nan
    temp_v[temp_v>300]=np.nan
    temp_v[temp_v<-300]=np.nan
        
    Tall.U=np.nanmean(temp_u, axis=0)
    Tall.V=np.nanmean(temp_v, axis=0)
    return Tall

#%% calculate sum of the average 
def interp(xi, yi, apriori_data, xf, yf, RESOLUTION=1/12):
    old_res=abs(xi[-1]-xi[-2])
    nan_indices = np.isnan(apriori_data)
    apriori_data = np.nan_to_num(apriori_data)
    s = RectBivariateSpline(xi, yi, apriori_data)
    posteriori_data = s(xf, yf)
    for idx, row in enumerate(nan_indices):
            for idy, was_nan in enumerate(row):
                if was_nan:
                    start_x = np.where(xf==xi[idx]) 
                    end_x = np.where(xf==xi[idx]+old_res) if idx !=len(xi)-1 else start_x
                    start_y = np.where(yf==yi[idy]) 
                    end_y =  np.where(yf==yi[idy]+old_res) if idy !=len(yi)-1 else start_y
                    for x_scaled in range(int(start_x[0]), int(end_x[0]), 1):
                        for y_scaled in range(int(start_y[0]), int(end_y[0]), 1):
                            posteriori_data[x_scaled, y_scaled] = np.NaN
    return posteriori_data.T

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

              
#%% plotting climatology
def draw_screen_poly(ax, lats, lons, fc_color, edge_color, linewidth=0.8,
                     zorder=3, projection=ccrs.PlateCarree(), alpha=0.7):
    xy = zip(lons,lats)
    poly = Polygon( list(xy), facecolor=fc_color, edgecolor=edge_color, 
                   alpha=alpha, linewidth=linewidth, zorder = zorder, transform=projection)
    ax.add_patch(poly)


def setmap_subplots_new(fig, axi, i, projection=ccrs.PlateCarree(), fc_color='blue', edge_color='blue'):
    
    coords={1: {'name': 'Azores_west',
                'lats': [36.80, 39.80],
                'lons': [-31.40, -25.00]},
            2: {'name': 'Madeira',
                'lats': [32.35, 33.15],
                'lons': [-17.55, -16.25]},
            3: {'name': 'Canaries',
                'lats': [27.50, 29.26],
                'lons': [-18.25, -13.40]},
            4: {'name': 'CV',
                'lats': [14.70, 17.30],
                'lons': [-25.40, -22.60]}}
        
    ax = fig.add_subplot(axi, projection=projection)
    if projection==ccrs.PlateCarree() or projection==ccrs.Mercator():
        gl = ax.gridlines(crs=projection, draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=100)
        gl.ylabels_left = True if (i+1) % 2 !=0 else False
        gl.xlabels_bottom = True 
        gl.xlabels_top = gl.ylabels_right = False 
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlocator = mticker.FixedLocator(range(-100, 60, 20))
    else:
        gl = ax.gridlines()
    ax.set_extent([-98, 35, 0, 47])
    
    # coast= cfeature.GSHHSFeature(scale='high',facecolor='0.8')
    # ax.add_feature(coast, edgecolor='black', linewidth=0.4, zorder=1)       
    borders=cfeature.BORDERS
    coast = NaturalEarthFeature(category='physical', scale='50m',
                            facecolor='none', name='coastline')
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        facecolor='0.9')
    ax.add_feature(land ,zorder=4)
    ax.add_feature(coast, edgecolor='gray', linewidth=0.5, zorder=4)
    ax.add_feature(borders, linewidth=0.3, zorder=4)
    for kk in coords.keys():
        lats=coords[kk]['lats'] + coords[kk]['lats'][::-1]
        mylist = (coords[kk]['lons'] * 2)
        myorder = [0, 2, 1, 3]
        lons = [mylist[i] for i in myorder]
        draw_screen_poly(ax, lats, lons,fc_color, edge_color, linewidth=1.5)
    return ax

def setmap_subplots(fig, i, projection=ccrs.PlateCarree(), fc_color='red', edge_color='red'):
    
    coords={1: {'name': 'Azores_west',
                'lats': [36.80, 39.80],
                'lons': [-31.40, -25.00]},
            2: {'name': 'Madeira',
                'lats': [32.35, 33.15],
                'lons': [-17.55, -16.25]},
            3: {'name': 'Canaries',
                'lats': [27.50, 29.26],
                'lons': [-18.25, -13.40]},
            4: {'name': 'CV',
                'lats': [14.70, 17.30],
                'lons': [-25.40, -22.60]}}
        
    ax = fig.add_subplot(2, 2, i+1, projection=projection)
    if projection==ccrs.PlateCarree() or projection==ccrs.Mercator():
        gl = ax.gridlines(crs=projection, draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                  zorder=100)
        gl.ylabels_right = True if (i+1) % 2 !=0 else False
        gl.xlabels_bottom = True if i<2 else False
        gl.xlabels_top = gl.ylabels_left = False 
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlocator = mticker.FixedLocator(range(-100, 60, 20))
    else:
        gl = ax.gridlines()
    ax.set_extent([-98, 35, 0, 48])
    
    # coast= cfeature.GSHHSFeature(scale='high',facecolor='0.8')
    # ax.add_feature(coast, edgecolor='black', linewidth=0.4, zorder=1)       
    borders=cfeature.BORDERS
    coast = NaturalEarthFeature(category='physical', scale='50m',
                            facecolor='none', name='coastline')
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        facecolor='0.85')
    ax.add_feature(land ,zorder=4)
    ax.add_feature(coast, edgecolor='gray', linewidth=0.5, zorder=4)
    ax.add_feature(borders, linewidth=0.3, zorder=4)
    for kk in coords.keys():
        lats=coords[kk]['lats'] + coords[kk]['lats'][::-1]
        mylist = (coords[kk]['lons'] * 2)
        myorder = [0, 2, 1, 3]
        lons = [mylist[i] for i in myorder]
        draw_screen_poly(ax, lats, lons,fc_color, edge_color, linewidth=1.5)
    return ax

#  Varying line width along a streamline
def plot_mesh(Tall, ax, cmap, stline_dens=1.5, stream_color='black', 
              projection=ccrs.PlateCarree(), colorbar=False):
    U=Tall.U
    V=Tall.V
    speed = np.sqrt(U**2 + V**2)
    xs, ys = np.meshgrid(Tall.lon, Tall.lat)
    pcolor=ax.pcolormesh(xs,ys,speed, cmap=cmap, transform=projection)
#    lw = 5*speed / np.nanmax(speed)
#    ax.streamplot(xs, ys, U, V, density=1.5, transform=projection,
#                         color='k', linewidth=lw)
    if stline_dens > 0:
        ax.streamplot(xs, ys, U, V, density=stline_dens, transform=projection,
                             color=stream_color)
    return pcolor

def add_colorbar_new(ax, co, i, cb_extend=None):
    location = 'top' if i == 0 else 'bottom' 
    size="3%" if i == 0 else "4.5%"
    if i == 0:
        divider = make_axes_locatable(ax)
        ax_cb = divider.append_axes(location, size=size, pad=0.02, axes_class=plt.Axes)
    else:
        ax_cb = fig.add_axes([ax.get_position().x0,
                              ax.get_position().y0-0.045,
                              ax.get_position().x1-ax.get_position().x0,
                              0.015]) 
    cb=fig.colorbar(co, cax=ax_cb, extend=cb_extend, orientation='horizontal')
    cb.set_label("Speed (m.s$^{-1}$)")
    ax_cb.xaxis.set_ticks_position(location)
    ax_cb.xaxis.set_label_position(location)
    
def add_colorbar(ax, co, i, cb_extend=None):
    location = 'left' if (i+1) % 2 else 'right' 
    divider = make_axes_locatable(ax)
    ax_cb = divider.append_axes(location, size="4%", pad=0.1, axes_class=plt.Axes)
    cb=fig.colorbar(co, cax=ax_cb, extend=cb_extend)
    cb.set_label("m.s$^{-1}$")
    ax_cb.yaxis.set_ticks_position(location)
    ax_cb.yaxis.set_label_position(location)

def add_text(textstr):
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax[-1].text(0.99, 0.02, textstr, transform=ax[-1].transAxes, 
                fontsize=12, horizontalalignment='right', verticalalignment='bottom', 
                bbox=props, zorder=5000)


#%% good things
#pre_path='/mnt/c/Cenas/Academico/OOM/CleanAtlantic/OceanParcels/nc/'
#start_date=datetime(2010,12,1)
#finish_date=datetime(2010,12,29)

if socket.gethostname().find('ciimar') == -1:
    pre_path='/media/claudio_nas/MERCATOR_data/' 
else:
    pre_path='/media/claudio_nas/MERCATOR_data/'
        
start_date=datetime(2006,1,1)
# finish_date=datetime(2006,1,29)
finish_date=datetime(2015,12,29)

data_type = {'MERCATOR_0083': pre_path + 'MERCATOR_0083res/',
             'MERCATOR_025': pre_path + 'MERCATOR_025res/',
             'stokes': pre_path + 'WW3_ECMWF/',
             'wind': pre_path + 'WIND_ECMWF/'}

Tall = {}
for i, s in enumerate(data_type):
    print('Reading "%s"' %s)
    Tall[s]=loading_data(data_type[s], start_date, finish_date)
    
isnan=np.where((Tall['wind'].U==1) & (Tall['wind'].V==1))
Tall['wind'].U[isnan]=np.nan
Tall['wind'].V[isnan]=np.nan

for i, s in enumerate(Tall):
    Tall[s].lon_interp=np.around(np.arange(-98, 40.0001, 1/12),decimals=4)
    Tall[s].lat_interp=np.around(np.arange(0, 70.0001, 1/12), decimals=4)
    if s !='MERCATOR_0083':
        Tall[s].U_interp=interp(Tall[s].lon, Tall[s].lat, Tall[s].U.T, 
            Tall[s].lon_interp, Tall[s].lat_interp)
        Tall[s].V_interp=interp(Tall[s].lon, Tall[s].lat, Tall[s].V.T, 
            Tall[s].lon_interp, Tall[s].lat_interp)
    else:
        Tall[s].U_interp=np.zeros([Tall[s].lat_interp.shape[0], Tall[s].lon_interp.shape[0]])
        Tall[s].V_interp=np.zeros([Tall[s].lat_interp.shape[0], Tall[s].lon_interp.shape[0]])
        _, lonx_ind,_ = np.intersect1d(Tall[s].lon_interp, Tall[s].lon,
                                          return_indices=True)
        _, latx_ind,_ = np.intersect1d(Tall[s].lat_interp, Tall[s].lat,
                                          return_indices=True)
        Tall[s].U_interp[latx_ind[0]:latx_ind[-1]+1,lonx_ind[0]:lonx_ind[-1]+1]=Tall[s].U
        Tall[s].V_interp[latx_ind[0]:latx_ind[-1]+1,lonx_ind[0]:lonx_ind[-1]+1]=Tall[s].V

lon_M_025=Tall['MERCATOR_025'].lon_interp
lat_M_025=Tall['MERCATOR_025'].lat_interp
kk_lon=np.where((lon_M_025 > np.min(Tall['MERCATOR_0083'].lon)) & (lon_M_025 < np.max(Tall['MERCATOR_0083'].lon)))
kk_lat=np.where((lat_M_025 > np.min(Tall['MERCATOR_0083'].lat)) & (lat_M_025 < np.max(Tall['MERCATOR_0083'].lat)))
Tall['MERCATOR_025'].U_interp[kk_lat[0][0]:kk_lat[0][-1]:, kk_lon[0][0]:kk_lon[0][-1]]=0
Tall['MERCATOR_025'].V_interp[kk_lat[0][0]:kk_lat[0][-1]:, kk_lon[0][0]:kk_lon[0][-1]]=0

total=ncData()
total.lat=Tall[s].lat_interp
total.lon=Tall[s].lon_interp
total.U = Tall['MERCATOR_025'].U_interp + Tall['MERCATOR_0083'].U_interp +\
                  (0.01*Tall['wind'].U_interp) + Tall['stokes'].U_interp
total.V = Tall['MERCATOR_025'].V_interp + Tall['MERCATOR_0083'].V_interp +\
                  (0.01*Tall['wind'].V_interp) + Tall['stokes'].V_interp
                  
#%% actual plot
import cmocean 
import matplotlib.gridspec as gridspec

cmap1=copy(plt.get_cmap('cmo.speed'))
# cmap1.set_over('orangered')
# plt.ioff()
lat_m = [12, 60, 60, 12] 
lon_m = [-33, -33, 10, 10]

ax=[]
co=[]

# plt.ioff()
fig=plt.figure(figsize=(13, 10), constrained_layout=False)
gs = gridspec.GridSpec(2, 2, wspace=0.06, hspace=0.0001, height_ratios=[2.5, 1.2], 
                        left=0.12, right=0.88, top=0.96, bottom=0.08)

i=0
ax.append(setmap_subplots_new(fig, gs[0:1, :], i))
co.append(plot_mesh(Tall['MERCATOR_025'], ax[-1], cmap1, stline_dens=3.5, 
                    stream_color='dimgray'))
co.append(plot_mesh(Tall['MERCATOR_0083'], ax[-1], cmap1, stline_dens=0))
draw_screen_poly(ax[-1], lat_m, lon_m, fc_color='none', edge_color='black', linewidth=1.5,
                 zorder=5, alpha=1)
imax = np.max([n.get_clim()[1] for n in co ])
for n in co:
    n.set_clim(vmin=0, vmax=0.6)
add_colorbar_new(ax[-1], co[-1], i, cb_extend='max')
textstr = 'Ocean currents'
add_text(textstr)

i=2
ax.append(setmap_subplots_new(fig, gs[1, 0], i))
co.append(plot_mesh(Tall['wind'], ax[-1], cmap1))
co[-1].set_clim(vmin=0)
add_colorbar_new(ax[-1], co[-1], i)
textstr = 'Wind'
add_text(textstr)

i=3
ax.append(setmap_subplots_new(fig, gs[1, 1], i))
co.append(plot_mesh(Tall['stokes'], ax[-1], cmap1))
co[-1].set_clim(vmin=0)
add_colorbar_new(ax[-1], co[-1], i)
textstr = '\n'.join((
    'Stokes',
    'Drift'))
add_text(textstr)

plt.savefig('/home/ccardoso/Cenas/Academico/OOM/CleanAtlantic/Papers/Part_1_auxilliary_files/climatology_new.png', 
            dpi=300)
plt.close()

# #%% old plot with 4 subplots
# fig=plt.figure(figsize=(14, 8))
# ax=[]
# co=[]


# i=0
# ax.append(setmap_subplots(fig, i, projection))
# co.append(plot_mesh(Tall['MERCATOR_025'], ax[-1], cmap1))
# co.append(plot_mesh(Tall['MERCATOR_0083'], ax[-1], cmap1))
# draw_screen_poly(ax[-1], lat_m, lon_m, fc_color='none', edge_color='yellow', linewidth=1.5,
#                  zorder=5)
# imax = np.max([n.get_clim()[1] for n in co ])
# for n in co:
#     n.set_clim(vmin=0, vmax=0.6)
# add_colorbar(ax[-1], co[-1], i, cb_extend='max')
# textstr = 'Currents'
# add_text(textstr)

# i=1
# ax.append(setmap_subplots(fig, i, projection))
# co.append(plot_mesh(Tall['wind'], ax[-1], cmap1))
# add_colorbar(ax[-1], co[-1], i)
# textstr = 'Wind'
# add_text(textstr)

# i=2
# ax.append(setmap_subplots(fig, i, projection))
# co.append(plot_mesh(Tall['stokes'], ax[-1], cmap1))
# add_colorbar(ax[-1], co[-1], i)
# textstr = '\n'.join((
#     'Stokes',
#     'Drift'))
# add_text(textstr)

# i=3
# ax.append(setmap_subplots(fig, i, projection)) 
# co.append(plot_mesh(total, ax[-1], cmap1))
# co[-1].set_clim(vmin=0, vmax=0.6)
# add_colorbar(ax[-1], co[-1], i, cb_extend='max')
# textstr = '\n'.join((
#     'Currents',
#     'Wind (1%)',
#     'Stokes Drift'))
# add_text(textstr)

# for j, v in enumerate(ax):
#     yoffset=0 if j<2 else 0
#     xoffset=-0.078 if (j+1) % 2 else -0.026
#     pos1 = v.get_position() # get the original position 
#     v.set_position([pos1.x0+xoffset , pos1.y0+yoffset,  pos1.width *1.25, pos1.height *1.25])
    
# plt.savefig('/home/ccardoso/Cenas/Academico/OOM/CleanAtlantic/Papers/Part_1/images/climatology_1.png', 
#         dpi=300, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format=None,
#         transparent=False, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)
# #plt.savefig('/home/ccardoso/c/Cenas/Academico/OOM/CleanAtlantic/OceanParcels/climatology', dpi=300, facecolor='w', edgecolor='w',
# #        orientation='portrait', papertype=None, format=None,
# #        transparent=False, bbox_inches=None, pad_inches=0.1,
# #        frameon=None, metadata=None)
# # plt.close()

