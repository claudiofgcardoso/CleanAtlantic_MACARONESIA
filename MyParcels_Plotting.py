#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:00:29 2020

@author: ccardoso
"""

import os
# os.chdir('/home/ccardoso/Cenas/Academico/OOM/CleanAtlantic/OceanParcels/My_scripts/CleanAtlantic_Madeira/')
print("Current working directory is:", os.getcwd() ) 
import sys
sys.path.append("..")
from datetime import datetime
import socket 
from argparse import ArgumentParser
import csv
import math
from netCDF4 import Dataset
from scipy import stats
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.axes as maxes
from matplotlib.patches import Polygon
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
from datetime import timedelta
import numpy as np
from os import environ
environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from copy import copy
import cmocean 
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import time
import pandas as pd
from netCDF4 import Dataset, num2date
import shapefile
from glob import glob

#%% loading data

def get_pop_dens():
    pop_raster_file='../Files/gpw_v4_population_density_rev11_2020_2pt5_min.asc'
    with open(pop_raster_file, 'r') as prism_f:
        prism_header = prism_f.readlines()[:6]
    
    # Read the PRISM ASCII raster header
    prism_header = [item.strip().split()[-1] for item in prism_header]
    prism_cols = int(prism_header[0])
    prism_rows = int(prism_header[1])
    prism_xll = float(prism_header[2])
    prism_yll = float(prism_header[3])
    prism_cs = float(prism_header[4])
    prism_nodata = float(prism_header[5])
    
    # Read in the PRISM array
    prism_array = np.loadtxt(pop_raster_file, dtype=np.float, skiprows=6)
     
    # Set the nodata values to nan
    prism_array[prism_array == prism_nodata] = np.nan
    
    # Calculate the extent of the PRISM array
    # Extent is Xmin, Xmax, Ymin, Ymax
    prism_extent = [prism_xll, prism_xll + prism_cols * prism_cs,
                    prism_yll, prism_yll + prism_rows * prism_cs]
    return prism_extent, prism_array

def get_fishing_hours(res=0.1):
    fish_files=sorted(glob('../Files/fishing_10_byyear_*.csv'))
    for i, file in enumerate(fish_files):
        if i == 0:
            fish_table=pd.read_csv(file)
        else:
            fish_table=fish_table.append(pd.read_csv(file))
    
    fish_table['lat_bin']*=0.1
    fish_table['lon_bin']*=0.1
    
    # bins=[np.unique(fish_table['lon_bin'].to_numpy()), np.unique(fish_table['lat_bin'].to_numpy())]
    bins=[np.arange(-180,180,res), np.arange(-90,90,res)]
    fish_matrix, xe, ye,_ = stats.binned_statistic_2d(fish_table['lon_bin'].to_numpy(), 
                                             fish_table['lat_bin'].to_numpy(),
                                             fish_table['fishing_hours'].to_numpy(), 
                                             bins=bins, 
                                             statistic='sum')
    xb = (xe[1:] + xe[:-1])/2
    yb = (ye[1:] + ye[:-1])/2
    X, Y = np.meshgrid(xb, yb)
    fish_matrix[0,: ] = fish_matrix[-1, :]
    
    return X, Y, fish_matrix

def load_particles_file(fname, varnames, start_date=None, finish_date=None, release=None,
                        organize_ocean=False, eliminate_beaching=False, calculate_age=True):
    class ParticleData(object):
        def __init__(self):
            self.trajectory = []
            self.loc = []
            
    def def_organize_ocean(T):
        interval=int(T.loc.shape[1]/47)
        for hh in range(0, T.loc.shape[1], (47*100)):
            for jjii, jj in enumerate(range(hh, hh+(47*100), interval)):
                T.loc[:,hh+jj:hh+jj+400]=jjii
        return T
        
    print('Loading file: %s' %fname)
    pfile = Dataset(fname, 'r')
    
    nc_time_units=pfile.variables['time'].units
    time_origin = num2date(0, nc_time_units)
    
    if start_date is not None and finish_date is not None:
        first_times=pfile.variables['time'][:,0][:]
        start_date_seconds=(start_date-time_origin).total_seconds()
        finish_date_seconds=(finish_date-time_origin).total_seconds()
        time_index=np.where((first_times>=start_date_seconds) & (first_times <= finish_date_seconds))[0]
    elif start_date is not None:
        first_times=pfile.variables['time'][:,0][:]
        start_date_seconds=(start_date-time_origin).total_seconds()
        time_index=np.where(first_times>=start_date_seconds)[0]
    elif finish_date is not None:
        first_times=pfile.variables['time'][:,0][:]
        finish_date_seconds=(finish_date-time_origin).total_seconds()
        time_index=np.where(first_times <= finish_date_seconds)[0]
    else:
        time_index=np.arange(0, pfile.variables['time'].shape[0])
        # time_index=np.arange(0,np.nanmax(pfile.variables['time'])+1, 
        #                      abs(pfile.variables['time'][:][0,1]-pfile.variables['time'][:][0,0]))
    
    loc=pfile.variables['loc'][:]
    if release is None:
        unique_loc=np.unique(loc[~np.isnan(loc)])        
        release = dict(enumerate(unique_loc.flatten(), 1))
    
    Tall = {}
    for s in release.keys():
        start = time.time()
        j_release = np.where(loc==s)[0]
        j=np.intersect1d(time_index,j_release)
        
        if len(j) >0: 
            print('Loading local "%s"' %release[s])
            Tall[s]=ParticleData()
            
            deleted_values=np.ma.getmask(pfile.variables['lon'][j,:])
                    
            for iv, v in enumerate(varnames):                    
                print('Reading variable "%s" '%v)
                
                if len(pfile.variables[v].shape) == 1:
                    setattr(Tall[s], v, np.reshape(np.repeat(pfile.variables[v][j], 
                                                                 pfile.variables['trajectory'][j,:].shape[1]), 
                                                       pfile.variables['trajectory'][j,:].shape)[~deleted_values].flatten())
                elif v == 'time' and calculate_age is True:
                    var_time=pfile.variables[v][j,:]
                else:
                    setattr(Tall[s], v, pfile.variables[v][j,:][~deleted_values].flatten())
                
                if eliminate_beaching:
                    index_beached=np.where(Tall[s].beached.data == 1)
                    Tall[s].lon.data[index_beached]=np.NaN
                    Tall[s].lat.data[index_beached]=np.NaN
                    Tall[s].age.data[index_beached]=np.NaN
    
            Tall[s].time_origin = time_origin
            if calculate_age is True:
                Tall[s].time=var_time[~deleted_values].flatten()
                Tall[s].age=np.concatenate((np.array([np.repeat(0, var_time.shape[0])]).T,
                                            np.cumsum(abs(np.diff(var_time, axis=1)), axis=1)),
                                           axis=1)[~deleted_values].flatten()
            
            end = time.time()
            period=end-start
            print('Completed -> Took "%g" to load' %period)
        
    if organize_ocean is True:
        Tall=def_organize_ocean(Tall[0])
    
    return Tall


def get_coastline(file_shp_buffer, file_shp_coast=None, stype='ROMS'):
    buffer_sf = shapefile.Reader(file_shp_buffer)
    buffer_records = buffer_sf.records()
    buffer_shapes = buffer_sf.shapes()
    
    if 'eez_v11_clipped.shp' in file_shp_buffer:
        index_key=5
    else:
        index_key=4
        
    coast_buffer={}
    for ind, s in enumerate(buffer_shapes):          
        if stype!='ROMS':
            coast_id = buffer_records[ind][index_key]
            lon1=[]
            lat1=[]
            for i in range(len(s.parts)):
                i_start = s.parts[i]
                if i==len(s.parts)-1:
                    i_end = len(s.points)
                else:
                    i_end = s.parts[i+1]
                lon1.append([i[0] for i in s.points[i_start:i_end]])
                lat1.append([i[1] for i in s.points[i_start:i_end]])
        else:
            coast_id = int(buffer_records[ind][4])
            lon1=np.array([point[0]+0.005 for point in s.points])
            lat1=np.array([point[1]-0.0034 for point in s.points])
        
        coast_buffer.update({coast_id: {'lat': lat1, 'lon': lon1, 'count': 0 ,
                                        'density': np.nan, 'len': 0}})
    
    if file_shp_coast is not None:
        coast_sf = shapefile.Reader(file_shp_coast)
        coast_records = coast_sf.records()
        coast_shapes = coast_sf.shapes()
        lon=[]
        lat=[]
        coast_id=[]
        for ind, s in enumerate(coast_shapes):
            if stype!='ROMS':
                lon.extend([point[0] for point in s.points])
                lat.extend([point[1] for point in s.points])
                cid=coast_records[ind][4]
            else:
                lon.extend([point[0]+0.005 for point in s.points])
                lat.extend([point[1]-0.0034 for point in s.points])
                cid=int(coast_records[ind][0])
            
            coast_id.extend([cid] * len(s.points))
            coast_buffer[cid]['len']=coast_records[ind][3]
        
        data_pd=pd.DataFrame(np.array([coast_id, lat, lon]).T, 
                              columns=['id','lat', 'lon'])
        # data_pd['id']=data_pd['id'].astype(int)
        data_pd['lat']=data_pd['lat'].astype(float)
        data_pd['lon']=data_pd['lon'].astype(float)
    else:
        data_pd=None
    return coast_buffer, data_pd

def calculate_distance(lon, lat, lon_r, lat_r):
    if ~np.isnan(lon_r) and ~np.isnan(lat_r):
        R = 6373.0
        lon1=np.radians(lon)
        lat1=np.radians(lat)
        lon2=np.radians(lon_r)
        lat2=np.radians(lat_r)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance=(R*c)
    else:
        distance=np.nan
    return distance
#%% functions to plot

def draw_screen_poly(ax, lons, lats, fc_color, edge_color=None, linewidth=0.8, zorder=1, alpha=0.7):
    if edge_color is None:
        edge_color=fc_color
    xy = zip(lons,lats)
    poly = Polygon(list(xy), facecolor=fc_color, edgecolor=edge_color, 
                   alpha=alpha, linewidth=linewidth, zorder = zorder)
    ax.add_patch(poly)
    
def add_release_locs(ax, Tall):
    for i, s in enumerate(Tall):
        ax.plot(Tall[s].lon[0], Tall[s].lat[0], 'o', color='none', 
                        ms=4, zorder=100, markeredgecolor='k',markeredgewidth=0.6)
    
def add_text(ax, textstr, fsize=12, position = 'southeast', x=None, y=None):
    # left, width = .25, .5
    # bottom, height = .25, .5
    # right = left + width
    # top = bottom + height
    if position == 'northwest':
        # x=left
        # y=top
        x=0.02
        y=0.98
        horizontalalignment='left'
        verticalalignment='top'
    elif position == 'southeast':
        # x=right
        # y=bottom
        x=0.98
        y=0.15
        horizontalalignment='right' 
        verticalalignment='bottom'
    elif position == 'southwest':
        # x=left
        # y=bottom
        x=0.02 if x is None else x
        y=0.03 if y is None else y
        horizontalalignment='left' 
        verticalalignment='bottom'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    ax.text(x, y, textstr, transform=ax.transAxes, 
                fontsize=fsize, horizontalalignment=horizontalalignment, 
                verticalalignment=verticalalignment, 
                bbox=props, zorder=3000)

def setmap_subplots(release, vertical = True, figsize=None, projection=ccrs.PlateCarree(), fc_color='red', 
                    edge_color=None, extent=[-98, 30, 0, 50], plot_release=False, inline_plotting=False,
                    pop_density=False, new_config=False, fishing=False):
    """
    function to generate subplots in a figure
    "vertical" can be: True, False or "divided" (for beaching)
    """
    SMALL_SIZE = 12
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15
    
    if vertical is True:
        figsize=(8.3,11.7) if figsize is None else figsize
        nrows=len(release.keys())
        ncols=1
        constrained_layout=True
    elif vertical is False:
        if pop_density is True or new_config==True:
            figsize=(14, 9) 
        elif figsize is None:
            figsize=(14, 8) 
        nrows=round(len(release.keys())/2)
        ncols=2
        constrained_layout=False
    elif vertical == 'divided':
        figsize=(11, 10) 
        nrows=round(len(release.keys())/2)
        ncols=4
        constrained_layout=False
    
    if vertical == 'divided':
        alpha_grid=0
        facecolor_coast='0.9' if pop_density is False else 'none'
        border_color='gray' if pop_density is False else 'black'
        border_witdh=0.6
    elif pop_density is True or new_config is True:
        alpha_grid=0 if fishing is True else 0.5
        facecolor_coast=cfeature.COLORS['land'] if fishing is True else '0.85'
        extent=[-100, 20, 0, 54]
        # extent=[-100, 20, 0, 50]
        border_color='gray'
        border_witdh=0.6
    else:
        alpha_grid=0.5
        facecolor_coast='0.85'
        border_color='gray'
        border_witdh=0.3
        
    borders=cfeature.BORDERS
    coast= cfeature.GSHHSFeature(scale='high')
    
    if pop_density is True:
        pop_raster_file='../Files/gpw_v4_population_density_rev11_2020_2pt5_min.asc'
        with open(pop_raster_file, 'r') as prism_f:
            prism_header = prism_f.readlines()[:6]
        
        # Read the PRISM ASCII raster header
        prism_header = [item.strip().split()[-1] for item in prism_header]
        prism_cols = int(prism_header[0])
        prism_rows = int(prism_header[1])
        prism_xll = float(prism_header[2])
        prism_yll = float(prism_header[3])
        prism_cs = float(prism_header[4])
        prism_nodata = float(prism_header[5])
        
        # Read in the PRISM array
        prism_array = np.loadtxt(pop_raster_file, dtype=np.float, skiprows=6)
         
        # Set the nodata values to nan
        prism_array[prism_array == prism_nodata] = np.nan
        
        # Calculate the extent of the PRISM array
        # Extent is Xmin, Xmax, Ymin, Ymax
        prism_extent = [prism_xll, prism_xll + prism_cols * prism_cs,
                        prism_yll, prism_yll + prism_rows * prism_cs]
        
    if inline_plotting is False:
        plt.ioff()
    else:
        plt.ion()
        
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                           subplot_kw=dict(projection=projection), 
                           constrained_layout=constrained_layout)
                 
    for n, axx in enumerate(ax.flatten()):
        if projection==ccrs.PlateCarree() or projection==ccrs.Mercator():
            gl = axx.gridlines(crs=projection, draw_labels=True,
                               linewidth=0.5, color='gray', alpha=alpha_grid, linestyle='--',
                               zorder=300)
            if vertical == 'divided':
                # gl.right_labels = True if n == 1 or n == 5 else False
                # gl.bottom_labels = True if n<4 else False
                # gl.top_labels = gl.left_labels = False 
                gl.ylabels_right = True if n == 1 or n == 5 else False
                gl.xlabels_bottom = True if n<4 else False
                gl.xlabels_top = gl.ylabels_left = False 
                # extent=[-99, -49, 0, 68] if n % 2 == 0 else [-33, 17, 0, 68]
                extent=[-99, -47, 0, 68] if n % 2 == 0 else [-36, 16, 0, 68]
                gl.ylocator = ticker.FixedLocator(range(0, 70, 10))
            elif vertical is True:
                # gl.left_labels = True 
                # gl.bottom_labels = True if n == len(ax)-1 else False
                # gl.top_labels = gl.right_labels = False
                gl.ylabels_left = True 
                gl.xlabels_bottom = True if n == len(ax)-1 else False
                gl.xlabels_top = gl.ylabels_right = False
            elif vertical is False:
                # gl.right_labels = True if n % 2 ==0 else False
                # gl.bottom_labels = True if n<2 else False
                # gl.top_labels = gl.left_labels = False 
                gl.ylabels_right = True if n % 2 ==0 else False
                gl.xlabels_bottom = True if n<2 else False
                gl.xlabels_top = gl.ylabels_left = False 
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlocator = ticker.FixedLocator(range(-100, 60, 20))
        else:
            gl = axx.gridlines()
        axx.set_extent(extent)
        axx.add_feature(coast, edgecolor='black', linewidth=0.3, zorder=3, facecolor=facecolor_coast)
        axx.add_feature(borders, linewidth=border_witdh, zorder=3, edgecolor=border_color)
        if vertical =='divided':
            ocean=cfeature.OCEAN
            axx.add_feature(ocean, facecolor='0.5')
            axx.add_feature(coast, edgecolor='none', zorder=1, facecolor='white')
        if (vertical == 'divided' and n % 2 == 0):
            add_text(axx, release[math.ceil(n/2)+1]['name'], BIGGER_SIZE,
                     position='southwest', x=0.04)
        elif vertical != 'divided':
            add_text(axx, release[n+1]['name'], BIGGER_SIZE, position='southwest')
        
        if plot_release is True:
            # for kk in release.keys():
            lats=release[n+1]['lats'] + release[n+1]['lats'][::-1]
            lons = [(release[n+1]['lons'] * 2)[i] for i in [0, 2, 1, 3]]
            draw_screen_poly(axx, lons, lats, fc_color, edge_color)
        
        if pop_density is True:
            img_plot = axx.imshow(prism_array, 
                                  extent=prism_extent,
                                  cmap='cmo.gray_r', 
                                  norm=colors.LogNorm(vmin=0.1, vmax=10000), 
                                  interpolation='nearest', 
                                  zorder=2)
            
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fon
    
    if fishing is True or new_config is True:
        for j, v in enumerate(ax.flatten()):
            yoffset=+0.02 if j<2 else 0.016
            xoffset=-0.08 if (j+1) % 2 else -0.024
            pos1 = v.get_position() # get the original position 
            v.set_position([pos1.x0+xoffset , pos1.y0+yoffset,  pos1.width *1.25, pos1.height *1.25])
            # if pop_density is True:
            #     add_colorbar(ax, img_plot, 'Population density (persons.km$^{-2}$)', location='top', fig=fig,
            #                  log=True, extend='max')
    elif vertical == 'divided':
        for j, v in enumerate(ax.flatten()):
            yoffset=-0.022 if j<4 else 0.04
            xoffset=+0.016 if (j == 2 or j == 3 or j == 6 or j == 7) else -0.08
            xoffset+=-0.027 if j % 2 == 0 else -0.0
            pos1 = v.get_position() # get the original position 
            v.set_position([pos1.x0+xoffset , pos1.y0+yoffset,  pos1.width *1.35, pos1.height *1.35])
        if pop_density is True:
            add_colorbar(ax, img_plot, 'Population density (persons.km$^{-2}$)', location='top', fig=fig,
                         log=True, extend='max')
        # cmap = plt.get_cmap('Spectral_r', lut=16)
        # cnorm=colors.Normalize(vmin=0, vmax=40)
        # co = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)   
        # add_colorbar(ax, co, 'test', location='bottom', fig=fig,
        #              log=True, extend='max')
    elif vertical is False:
        for j, v in enumerate(ax.flatten()):
            yoffset=0.04 if j<2 else 0.01
            xoffset=-0.08 if (j+1) % 2 else -0.024
            pos1 = v.get_position() # get the original position 
            v.set_position([pos1.x0+xoffset , pos1.y0+yoffset,  pos1.width *1.25, pos1.height *1.25])
    
    return fig, ax

def add_colorbar(ax, mappable, label, location='right', divider=None, extend='neither',
                 log = False, fig=None, ticks=None, spacing='proportional', mappable2=None):
    if type(ax) is not np.ndarray and type(ax) != list:
        if location == 'bottom':
            pad=0.4
            size="4%"
        elif location == 'left':
            pad=0.8
            size="3%"
        elif location == 'right':
            pad=0.1
            size="3%"
        elif location == 'top':
            pad=0.1
            size="4%"
        else:
            print('Invalid location!')
        
        if divider is None:
            divider = make_axes_locatable(ax)
            
        cax = divider.append_axes(location, size=size, pad=pad, axes_class=maxes.Axes)
        if location == 'bottom' or location == 'top':
            cb=plt.colorbar(mappable, cax=cax, orientation='horizontal', extend=extend, spacing=spacing) 
            if log is True:
                cb.ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
            cb.ax.set_xlabel(label)
            cb.ax.xaxis.set_label_position(location)
        elif location == 'left':
            cb=plt.colorbar(mappable, cax=cax, extend=extend, spacing=spacing) 
            if log is True:
                cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
            cb.ax.set_ylabel(label)
            cb.ax.yaxis.set_label_position(location)
            cb.ax.yaxis.tick_left()
        else:
            cb=plt.colorbar(mappable, cax=cax, extend=extend, spacing=spacing) 
            if log is True:
                cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
            cb.ax.set_ylabel(label)
            
        return divider
    
    elif type(ax) == list:
        cax = fig.add_axes([ax[10].get_position().x0,
                            ax[10].get_position().y0-0.02,
                            ax[15].get_position().x1-ax[10].get_position().x0,
                            0.015])
        
        if ticks == 'log':
            ticks = np.linspace(mappable.get_clim()[0],(mappable.get_clim()[0]*10)/2,5)
            ticks = np.concatenate((ticks,np.linspace(mappable.get_clim()[0]*10,(mappable.get_clim()[0]*100)/2,5)))
            ticks = np.concatenate((ticks,np.arange(mappable.get_clim()[0]*100,mappable.get_clim()[1]+1,10)))
            cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', spacing=spacing, 
                            extend=extend, ticks=ticks, format='%.1f')
        else:
            cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', extend=extend, spacing=spacing)
        cb.set_label(label)
        
    else:
        if ax.shape[0]*ax.shape[1] == 4:
            if location=='top':
                cax = fig.add_axes([ax.flatten()[0].get_position().x0,
                                    ax.flatten()[0].get_position().y1+0.006,
                                    ax.flatten()[1].get_position().x1-ax.flatten()[0].get_position().x0,
                                    0.025])
                
            else:
                if mappable2 is not None:
                    cax_top = fig.add_axes([ax.flatten()[2].get_position().x0,
                                    ax.flatten()[2].get_position().y0-0.026,
                                    ax.flatten()[3].get_position().x1-ax.flatten()[2].get_position().x0,
                                    0.02]) 
                    cax_bottom = fig.add_axes([ax.flatten()[2].get_position().x0,
                                       ax.flatten()[2].get_position().y0-0.046,
                                       ax.flatten()[3].get_position().x1-ax.flatten()[2].get_position().x0,
                                       0.02]) 
                else:
                    cax = fig.add_axes([ax.flatten()[2].get_position().x0,
                                        ax.flatten()[2].get_position().y0-0.028,
                                        ax.flatten()[3].get_position().x1-ax.flatten()[2].get_position().x0,
                                        0.025])  
        else:
            if location == 'top':
                cax = fig.add_axes([ax.flatten()[0].get_position().x0,
                                ax.flatten()[0].get_position().y1+0.005,
                                ax.flatten()[3].get_position().x1-ax.flatten()[0].get_position().x0,
                                0.02]) 
            else:
                cax = fig.add_axes([ax.flatten()[4].get_position().x0,
                                ax.flatten()[4].get_position().y0-0.025,
                                ax.flatten()[7].get_position().x1-ax.flatten()[4].get_position().x0,
                                0.02]) 
        # cb=fig.colorbar(mappable, ax=ax.flat, orientation='horizontal', pad=0.008, fraction=0.001)
        if log is True:   
            # cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', ticks=mappable.levels.tolist(), 
            #                 format=ticker.LogFormatter(), spacing=spacing)
            
            if ticks is None:
                # ticks=[10e0, 10e1, 10e2, 10e3, 10e4, 10e5]
                # ticks_format=ticker.LogFormatter(10, labelOnlyBase=False)
                cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', spacing=spacing, 
                            extend=extend, ticks=ticks, format='%.1f')
                cax.xaxis.set_ticks_position(location)
                cax.xaxis.set_label_position(location)
                
            elif ticks == 'log':
                ticks = np.linspace(mappable.get_clim()[0],(mappable.get_clim()[0]*10)/2,5)
                ticks = np.concatenate((ticks,np.linspace(mappable.get_clim()[0]*10,(mappable.get_clim()[0]*100)/2,5)))
                ticks = np.concatenate((ticks,np.arange(mappable.get_clim()[0]*100,mappable.get_clim()[1]+1,10)))
                
                if mappable2 is not None:
                    cb=fig.colorbar(mappable, cax=cax_bottom, orientation='horizontal', spacing=spacing, 
                                    extend=extend, ticks=ticks, format='%.1f')
                    cb.ax.tick_params(axis='x', which='both', direction='in')
                    cb_top=fig.colorbar(mappable2, cax=cax_top, orientation='horizontal', spacing=spacing, 
                                    extend=extend, ticks=cb.ax.get_xticks())
                    cb_top.ax.tick_params(axis='x', direction='in', which='both')
                    for ilabel in cb_top.ax.xaxis.get_ticklabels():
                        ilabel.set_visible(False)
                else:
                    cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', spacing=spacing, 
                                    extend=extend, ticks=ticks, format='%.1f')
                
                # cb.locator=ticker.LogLocator()
                # cb.formatter=ticker.LogFormatter()
            cb.set_label(label)
        else:
            cb=fig.colorbar(mappable, cax=cax, orientation='horizontal', extend=extend, spacing=spacing,
                            label=label)
            cax.xaxis.set_ticks_position(location)
            cax.xaxis.set_label_position(location)
            # cb.set_label(label)
            if type(ticks)==np.ndarray:
                cb.set_ticks(ticks)
            elif ticks =='years':
                t_ticks=np.arange(0, mappable.get_clim()[1], 365)
                t_ticks_labels=np.arange(0, len(t_ticks))
                cb.set_ticks(t_ticks)
                cb.set_ticklabels(t_ticks_labels)
        # cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    return cb

def set_max(co, imax=None, imin=None, method='max'):
    """
    Find maximum value between subplots
    """
    if imax is None:
        if method == 'second_largest':
            imax = np.sort([n.get_clim()[1] for n in co])[-2]
        if method == 'max':
            imax = np.max([n.get_clim()[1] for n in co])
        elif method == 'mean':
            imax = np.mean([n.get_clim()[1] for n in co])
    if imin is None:   
        imin= np.min([n.get_clim()[0] for n in co])
    for n in co:
        n.set_clim(vmin=imin, vmax=imax)
        

def plt_particles(fig, ax, Tall, plot_depth=False):
    scat = {}
    if plot_depth is False:
        for i, s in enumerate(Tall):
            b = Tall[s].time == 0
            scat[s] = ax.scatter(Tall[s].lon[b], Tall[s].lat[b], c=plt.cm.hsv(float(s)/len(Tall)), 
                                 s=5, alpha=.5, transform=ccrs.PlateCarree())
    else:
        for i, s in enumerate(Tall):
            b = Tall[s].time == 0
            scat[s]= ax.scatter(Tall[s].lon[b], Tall[s].lat[b], c=Tall[s].z[b], cmap=plt.cm.jet, 
                                s=5, alpha=.5, vmin=0, vmax=4500, transform=ccrs.PlateCarree()) 
    return scat
            
def plt_hist_particles(fig, ax, Tall, stype='ROMS', unique = False, beaching=False, buffer=False, 
                       percentage=False, density=False, plot_release=False, log=False, age=False,
                       hist_depth=False, hist2d_windage=False, method='mean', cmap_name='Spectral_r',
                       set_clip=None, imax=None):
    def count_unique(values):
        return np.unique(values).shape[0]
    def count_mode(values):
        return stats.mode(values)[0][0]
    def count_mode_unique(values):
        unique_p0_round=np.fix(np.unique(values)*100)+1 
        return stats.mode(unique_p0_round)[0][0]
        
    def deal_with_H(H, xe, ye):
        H[0,: ] = H[-1, :]
        xb = (xe[1:] + xe[:-1])/2
        yb = (ye[1:] + ye[:-1])/2
        xbb, ybb = np.meshgrid(xb, yb)
        xbb[xbb>180]=xbb[xbb>180]-360
        return H, xbb, ybb
    
    #creating colormaps
    if hist2d_windage is True and (method == 'mode' or method=='mode_unique'):
        cmap = copy(plt.get_cmap(cmap_name, 5))
    elif age is True:
        levels=round((imax/365)*4) if method!='std' else round((imax/365)*6)
        cmap = copy(plt.get_cmap(cmap_name, levels))
    else:
        cmap = copy(plt.get_cmap(cmap_name))
        cmap.set_under('white')      
        cmap.set_over('magenta')
        
    #loading data from particle file
    if stype == 'ROMS':
        nc_grid='../Files/roms_grid_d02_Zlev_Okean.nc'
        ROMS_grid=Dataset(nc_grid)
        
        nc_lon=ROMS_grid.variables['lon_psi'][0,:].data
        nc_lat=ROMS_grid.variables['lat_psi'][:,0].data
        # nc_lon=ROMS_grid.variables['lon_rho'][0,:].data
        # nc_lat=ROMS_grid.variables['lat_rho'][:,0].data
        nc_lon_grid, nc_lat_grid = np.meshgrid(nc_lon, nc_lat)
        bins=[nc_lon, nc_lat]
        
        shape_0=Tall[0].lon.shape[0]
        shape_1=0
        for s in Tall:
            shape_1+=Tall[s].lon.shape[1]
        
        lon=np.empty((shape_0, shape_1))
        lat=np.empty((shape_0, shape_1))
        if unique is True or beaching is True or age is True or hist2d_windage is True:
            weights=np.empty((shape_0, shape_1))
        if beaching is True:
            beached=np.empty((shape_0, shape_1))
            try:
                depth=np.empty((shape_0, shape_1))
            except:
                depth=None
            
        count=0
        for s in Tall:
            shape_n=Tall[s].lon.shape[1]
            lon[:,count:count+shape_n]=Tall[s].lon
            lat[:,count:count+shape_n]=Tall[s].lat
            if age is True:
                weights[:,count:count+shape_n]=np.abs(Tall[s].age)/86400
            elif unique is True or beaching is True:
                weights[:,count:count+shape_n]=Tall[s].trajectory
            elif hist2d_windage is True:
                if method=='mean':
                    weights[:,count:count+shape_n]=Tall[s].p0*100
                elif method=='mode':
                    weights[:,count:count+shape_n]=np.fix(Tall[s].p0*100)+1 
                elif method=='mode_unique':
                    weights[:,count:count+shape_n]=Tall[s].trajectory
            if beaching is True:
                beached[:,count:count+shape_n]=Tall[s].beached
                if depth is not None:
                    depth[:,count:count+shape_n]=Tall[s].z
                
            count+=shape_n
        
        # lon = lon.flatten()
        # lat = lat.flatten()
        # if age is True or unique is True or beaching is True:
        #     weights=weights.flatten()
        # if beaching is True:
        #     if depth is not None:
        #         depth=depth.flatten()
        #     beached=beached.flatten()
    
    elif stype=='MERCATOR': 
        #----> finish this condition for MERCATOR!
        nc_lon=np.arange(-100, 31, 0.5)
        nc_lat=np.arange(0, 70, 0.5)
        bins=[nc_lon, nc_lat]
        nc_lon_grid, nc_lat_grid = np.meshgrid(nc_lon, nc_lat)
                
        lon=Tall.lon
        lat=Tall.lat
        shape_1=len(np.unique(Tall.trajectory))
        # shape_1=lon.shape[1]
        if age is True:
            weights=np.abs(Tall.age)/86400
        elif unique is True or beaching is True:
            weights=Tall.trajectory
        elif hist_depth is True:
            weights=Tall.z
        elif hist2d_windage is True:
            if method=='mean':
                weights=Tall.p0*100
            elif method=='mode':
                weights=np.fix(Tall.p0*100)+1
            elif method=='mode_unique':
                weights=Tall.p0
                    
        if beaching is True:
            beached=Tall.beached
            try:
                depth=Tall.z
            except:
                depth=None
        
    else:
        print('WRONG stype!!')      

    if hist2d_windage is True:
        if method == 'mode':
            HH, xe, ye,_ = stats.binned_statistic_2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)].data,
                                                     weights[~np.isnan(lon)].data, bins=bins, 
                                                     statistic=count_mode)
        elif method=='mode_unique':
            HH, xe, ye,_ = stats.binned_statistic_2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)].data,
                                                     weights[~np.isnan(lon)].data, bins=bins, 
                                                     statistic=count_mode_unique)
        else:
            HH, xe, ye,_ = stats.binned_statistic_2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)].data,
                                                  weights[~np.isnan(lon)].data, bins=bins, 
                                                  statistic=method)
        H, xbb, ybb = deal_with_H(HH, xe, ye)
        
        co = ax.pcolormesh(xbb, ybb, H.T, 
                           cmap = cmap,
                           zorder = 0)
        
    elif age is True or hist_depth is True:
        HH, xe, ye,_ = stats.binned_statistic_2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)].data,
                                                 weights[~np.isnan(lon)].data, bins=bins, 
                                                 statistic=method)
        H, xbb, ybb = deal_with_H(HH, xe, ye)
        
        if method == 'std':
            H[np.where(H==0)]=np.nan
        co = ax.pcolormesh(xbb, ybb, H.T, 
                            cmap = cmap,
                            zorder = 0)
        # co = ax.contourf(xbb, ybb, H.T, levels=levels, zorder =0, extend='max')
        
    elif unique is True:
        HH, xe, ye,_ = stats.binned_statistic_2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)].data,
                                                 weights[~np.isnan(lon)].data, bins=bins, 
                                                 statistic=count_unique)
        H=(HH/shape_1)*100
        H = np.ma.masked_where(H < np.min(H)+1e-15, H)
        H, xbb, ybb = deal_with_H(H, xe, ye)
        
        cnorm=colors.LogNorm(vmin = 0.1, vmax = 30) if log is True else None
        
        if set_clip is None:
            co = ax.pcolormesh(xbb, ybb, H.T,   
                               norm = cnorm, 
                               cmap = cmap,
                               zorder = 0)
        else:
            cmap_gray = copy(plt.get_cmap('gray_r'))
            cmap_gray.set_under('white')      
            ax.pcolormesh(xbb, ybb, H.T,   
                          norm = cnorm, 
                          cmap = cmap_gray,
                          zorder = 0)
            for ipoly in set_clip:
                patch = PathPatch(ipoly, 
                                  facecolor='none', 
                                  linewidth=0.8,
                                  edgecolor='crimson')
                ax.add_patch(patch)
                co = ax.pcolormesh(xbb, ybb, H.T,   
                                   norm = cnorm, 
                                   cmap = cmap,
                                   zorder = 1,
                                   clip_path=patch)
        
    elif beaching is True:
        cc = (beached == 1)
        if depth is not None and stype!='MERCATOR':
            tt = (depth < 10)
            cc = np.logical_and(cc, tt)
        
        H, xe, ye,_ = stats.binned_statistic_2d(lon[cc], lat[cc].data,
                                                weights[cc].data, bins=bins, statistic=count_unique)
        H, xbb, ybb = deal_with_H(H, xe, ye)
        H=H.T
        if buffer is False:
            cnorm=colors.LogNorm() if log is True else None
            H = np.ma.masked_where(H < 10, H)   
            co= ax.pcolormesh(xbb, ybb, H,   
                                 norm = cnorm, 
                                 cmap = cmap,
                                 zorder = 20)
        else:
            nonzero=np.where(H>0)
            # testing
            # ax.plot(lon[cc], lat[cc], '.', color='red', ms=1, zorder=100)
            # ax.plot(xbb[nonzero], ybb[nonzero], '.', color='black', ms=5, zorder=110)
            
            #gettign shapes
            if stype=='ROMS':
                file_shp_buffer='../Files/AllIslands_long_buffer_splitted_concelhos_Dserta_merged.shp'
                file_shp_coast='../Files/Madeira_coastline_WGS84_split_concelhos.shp'
            else:
                raise Exception("Histogram (buffer) for beached particles on MERCATOR simulations \
                                must be done with 'plt_hist_particles_beached_buffer'")

            coast_buffer, coastline=get_coastline(file_shp_buffer, file_shp_coast, stype=stype)
            
            for i in range(nonzero[0].shape[0]):
                lon_p=xbb[0,nonzero[1][i]]
                lat_p=ybb[nonzero[0][i],0]
                pnumber=H[nonzero[0][i],nonzero[1][i]]
                
                dist=calculate_distance(coastline['lon'], coastline['lat'], lon_p, lat_p)
                JJ=np.unravel_index(np.argmin(dist, axis=None), dist.shape)[0]
                cid=coastline['id'][JJ]
                coast_buffer[cid]['count']+=int(pnumber)
            
            if density is True:
                for key, value in coast_buffer.items():
                    value['density']=value['count']/value['len']
                max_scale=np.max([value['density'] for key, value in coast_buffer.items()]) 
                min_scale=0

            else:
                if percentage is True:
                    sum_beached=np.sum([value['count'] for key, value in coast_buffer.items()]) 
                    for key, value in coast_buffer.items():
                        value.update(percentage= (value['count']/sum_beached)*100)
                    max_scale=np.max([value['percentage'] for key, value in coast_buffer.items()]) 
                    cnorm=colors.LogNorm(vmin=0.1, vmax=max_scale) if log is True else \
                          colors.Normalize(vmin=0, vmax=max_scale)
                else:
                    max_scale=np.max([value['count'] for key, value in coast_buffer.items()])  
                    cnorm=colors.LogNorm(vmin=1, vmax=max_scale) if log is True else \
                        colors.Normalize(vmin=1, vmax=max_scale)
                
            for pol in coast_buffer:
                if percentage is True:
                    color = cmap(cnorm(coast_buffer[pol]['percentage'])) 
                elif density is True:
                    color = cmap((coast_buffer[pol]['density']-min_scale)/max_scale)
                else:
                    color= cmap(cnorm(coast_buffer[pol]['count']))
                
                draw_screen_poly(ax, coast_buffer[pol]['lon'], coast_buffer[pol]['lat'], 
                                 color, alpha=1, zorder=0)

            co = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)    
            
    else:
        H, xe, ye = np.histogram2d(lon[~np.isnan(lon)], lat[~np.isnan(lat)], bins=bins)
        H, xbb, ybb = deal_with_H(H, xe, ye)
        if log is False:
            if stype=='ROMS':
                levels=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000] 
                # levels= np.arange(1, 10000, 1000)
            else:
                levels=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
                # levels= np.arange(1, 100000, 10000)
            co = ax.contourf(xbb, ybb, H.T, levels=levels, zorder =0, extend='max')
        else:

            # co = ax.pcolormesh(xbb, ybb, H.T, norm=colors.LogNorm(), zorder=0,
            #                    shading='nearest', cmap=plt.get_cmap('viridis', 12))
            
            lvls = np.linspace(10e0,10e1,10)
            lvls = np.concatenate((lvls[:-1],np.linspace(10e1,10e2,10)))
            lvls = np.concatenate((lvls[:-1],np.linspace(10e2,10e3,10)))
            lvls = np.concatenate((lvls[:-1],np.linspace(10e3,10e4,10)))
            co = ax.contourf(xbb, ybb, H.T, levels=lvls, norm = colors.LogNorm(), 
                              zorder =0, extend='max')
            co.cmap.set_over('orangered')
            
    if plot_release is True:
        add_release_locs(ax, Tall) 
      
    return co


def plt_hist_particles_beached_buffer(fig, ax_b, Tall, var='count', plot_release=False, log=False,
                                      vertical='divided', fname=None, method='mean', max_scale=40):
    """
    Function to calculate beached particles. ONLY WORKS FOR MERCATOR (subplots)
    var='count', 'percentage' or 'age'
    """
    def write_text_data(fname, Tall, var):
        with open(fname +'.csv', 'w', newline='') as csvfile:
            fieldnames = ['location', 'country']
            if var=='percentage':
                fieldnames.extend(['count', var])
            else:
                fieldnames.append(var)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for s, (key_tall, Tall_i) in enumerate(Tall.items()):
                for key_buffer, buffer in Tall_i.coast_buffer.items():
                    if var=='percentage':
                        writer.writerow({'location': key_tall, 
                                         'country': key_buffer,
                                         'count': buffer['count'],
                                         'percentage': buffer['percentage']})
                    else:
                        writer.writerow({'location': key_tall, 
                                         'country': key_buffer,
                                         var: buffer[var]})
                        
    def count_unique(values):
        return np.unique(values).shape[0]
        
    def deal_with_H(H, xe, ye):
        H[0,: ] = H[-1, :]
        xb = (xe[1:] + xe[:-1])/2
        yb = (ye[1:] + ye[:-1])/2
        xbb, ybb = np.meshgrid(xb, yb)
        xbb[xbb>180]=xbb[xbb>180]-360
        return H, xbb, ybb
    
    def draw_buffers(Tall, axi, var):
        for key_pol, pol in Tall.coast_buffer.items():
            if pol[var] > 0:
                color = cmap(cnorm(pol[var])) 
                if type(pol['lon'])==list:
                    for p in range(len(pol['lon'])): 
                        draw_screen_poly(axi, pol['lon'][p], pol['lat'][p], 
                                         color, alpha=1, zorder=0, 
                                         edge_color='black', linewidth=0.6)
                else:
                    draw_screen_poly(axi, pol['lon'], pol['lat'], 
                                     color, alpha=1, zorder=0, 
                                     edge_color='black', linewidth=0.6)
                    
    ax=ax_b.flatten()
    #creating colormaps
    
    if max_scale == 40:
        cmap = plt.get_cmap('Spectral_r', lut=16)
    else:
        cmap = plt.get_cmap('Spectral_r')
    # cmap.set_under('white')      
    
    nc_lon=np.arange(-100, 31, 0.5)
    nc_lat=np.arange(0, 70, 0.5)
    bins=[nc_lon, nc_lat]
    nc_lon_grid, nc_lat_grid = np.meshgrid(nc_lon, nc_lat)
    
    # file_shp_buffer='../Files/eez_cplitted_Dissolved_corrected.cpg'
    file_shp_buffer='../Files/eez_v11_clipped.shp'
    file_shp_coast='../Files/Lines_clipped_eez_12nm_v3.shp'
        
    for s, (key, Tall_i) in enumerate(Tall.items()):
        coast_buffer, coastline=get_coastline(file_shp_buffer, file_shp_coast, stype='MERCATOR')
        lon=Tall_i.lon.flatten()
        lat=Tall_i.lat.flatten()
        if var=='age':
            weights=Tall_i.age.flatten()
        else:
            weights=Tall_i.trajectory.flatten()

        beached=Tall_i.beached.flatten()
        # depth=Tall_i.z.flatten()
    
        cc = (beached == 1)
        # tt = (depth < 10)
        # cc = np.logical_and(cc, tt)
    
        if var != 'age':
            H, xe, ye,_ = stats.binned_statistic_2d(lon[cc], lat[cc].data,
                                                    weights[cc].data, bins=bins, statistic=count_unique)
        else:
            H, xe, ye,_ = stats.binned_statistic_2d(lon[cc], lat[cc].data,
                                                    weights[cc].data, bins=bins, statistic=method)
            
        H, xbb, ybb = deal_with_H(H, xe, ye)
        H=H.T

        nonzero=np.where(H>0)
        # testing
        # ax.plot(lon[cc], lat[cc], '.', color='red', ms=1, zorder=100)
        # ax.plot(xbb[nonzero], ybb[nonzero], '.', color='black', ms=5, zorder=110)
        
        if var=='age':
            for _, val in coast_buffer.items():
                val['pixels_count']=0
                val['sum_age']=0
                
        for i in range(nonzero[0].shape[0]):
            lon_p=xbb[0,nonzero[1][i]]
            lat_p=ybb[nonzero[0][i],0]
            pnumber=H[nonzero[0][i],nonzero[1][i]]
            
            dist=calculate_distance(coastline['lon'], coastline['lat'], lon_p, lat_p)
            JJ=np.unravel_index(np.argmin(dist, axis=None), dist.shape)[0]
            cid=coastline['id'][JJ]
            if var != 'age':
                coast_buffer[cid]['count']+=int(pnumber)
            else:
                coast_buffer[cid]['pixels_count']+=1
                coast_buffer[cid]['sum_age']+=pnumber
        
        if var =='age':
            for key, val in coast_buffer.items():
                val['age']=(val['sum_age']/val['pixels_count'])/86400 if val['sum_age'] > 0 else 0
                
        if var == 'percentage':
            sum_beached=np.sum([value['count'] for key, value in coast_buffer.items()]) 
            for key, value in coast_buffer.items():
                value.update(percentage= (value['count']/sum_beached)*100)

        Tall_i.coast_buffer=coast_buffer
        
    if max_scale ==0:
        for s, (_, Tall_i) in enumerate(Tall.items()):
            max_value=np.max([value[var] for _, value in Tall_i.coast_buffer.items()]) 
            max_scale=max_value if max_value > max_scale else max_scale
    
    cnorm=colors.LogNorm(vmin=1, vmax=max_scale) if log is True else \
                colors.Normalize(vmin=0, vmax=max_scale)
        
    count=0
    for s, (_, Tall_i) in enumerate(Tall.items()):
        if vertical == 'divided':
            for axi in ax.flatten()[count:count+2]:
                draw_buffers(Tall_i, axi, var)
            count+=2
        else:
            draw_buffers(Tall, ax[s], var)
               
    co = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)   
    
    if fname is not None:
        write_text_data(fname, Tall, var)
    
    return co


def plot_scatter_age_MERCATOR(Tall, release, plot_name, snapshot=None, extent=[-98, 30, 0, 50],
                              vertical=False, plot_release=False):
    fig, ax = setmap_subplots(release, vertical = vertical, plot_release=plot_release, 
                              extent=extent)
    
    co=[]
    for s, (_, Tall_i) in enumerate(Tall.items()):
        co.append(ax.flatten()[s].scatter(Tall_i.lon, Tall_i.lat, c=Tall_i.age/(24*3600), s=.1, 
                  cmap='jet', edgecolors='none', zorder=1, alpha=1))
    
    set_max(co)
    add_colorbar(ax, co[0], 'Particle age (years)', fig=fig, ticks='Years')
        
    plt.savefig(plot_name+'_scatter_age', dpi=300)
    plt.close(fig)

def plot_hist2d_MERCATOR(Tall, plot_name, release, plot_release=False, unique = False, 
                         beaching = False, hist_depth=False, hist2d_windage=False, percentage=False, 
                         buffer=False, log=False, age=False, method='mean', extent=[-98, 30, 0, 50],
                         inline_plotting=False, cb_extend='neither', cb_ticks=None,
                         imax=None, imin=None, cmap_name='Spectral_r', pop_density=False,
                         fishing_efforts=False, new_config=False):
    """
    This function produces an histogram map, with unique particle id or the total number os particles
    in a given pixel (standard).
    
    """
    if beaching is True:
        unique = False
        vertical='divided'
    else:
        vertical=False
        
    plot_name+='_histogram'
    
        
    if hist2d_windage is True:
        if method=='mean':
            plabel='Mean windage (%)'
            plot_name+='_windageMean' 
            imax=5
            imin=0
            cmap_name='cmo.thermal'
        elif method=='mode' or method=='mode_unique':
            plot_name+='_windageMode' if method=='mode' else '_windageMode_unique' 
            plabel='Windage class (mode)'
            imax=5.5
            imin=0.5
            cb_ticks=np.arange(1,6)
        cb_extend='neither'
            
    elif hist_depth is True:
        if method == 'median':
            plabel='Median depth (m)'
            plot_name+='_MedianDepth' 
            imax=20
        elif method=='mean':
            plabel='Mean depth (m)'
            plot_name+='_MeanDepth' 
            imax=40
        elif method == 'std':
            plabel='Standard deviation for particles depth (m)'
            plot_name+='_stdDepth' 
            imax=None
        elif method == 'max':
            plabel='Maximum particle depth (m)'
            plot_name+='_MaxDepth' 
            imax=None
        cmap_name='cmo.deep'
        imin=0
        cb_extend='max'
        
    elif unique is False and beaching is False and age is False:
        plabel='Number of particle occurrences'
        cb_extend='max'
        
    elif unique is True and beaching is False:
        plabel='Proportion of particles (%)'
        if log is False:
            plot_name+='_unique'
            imin=0.01
            imax=27  
            cb_extend='both'
        else:
            plot_name+='_unique_log'
            imin=0.1
            imax=30
            cb_extend='both'
            cb_ticks='log'
        if fishing_efforts is True:
            plot_name+='_fishing'
            
    elif beaching is True and buffer is False:
        plabel='Beached particles'
        plot_name+='_beaching' 
        
    elif beaching is True and buffer is True and percentage is False and age is False:
        plabel='Beached particles'
        plot_name+='_beaching_buffer'
        var='count'
        
    elif beaching is True and buffer is True and percentage is True:
        plabel='Proportion of beached particles (%)'
        plot_name+='_beaching_buffer_percentage'
        var='percentage'
        
    elif beaching is True and buffer is True and age is True:
        if method == 'median':
            plabel='Median beached particles age (years)'
            plot_name+='_beaching_buffer_MedianAge'
        elif method=='mean':
            plabel='Mean beached particles age (years)'
            plot_name+='_beaching_buffer_MeanAge'
        elif method == 'std':
            plabel='Standard deviation for beached particles age (years)'
            plot_name+='_beaching_buffer_StdAge'
        var='age'
        cb_ticks='years'
        
    elif beaching is False and age is True:
        # cmap_name='cmo.thermal'
        cb_extend='max'
        imin=0
        imax=(5*365)+1 if method != 'std' else (2*365)+1
        if method == 'median':
            plabel='Median particle age (years)'
            plot_name+='_MedianAge'
        elif method=='mean':
            plabel='Mean particle age (years)'
            plot_name+='_MeanAge'
        elif method == 'std':
            plabel='Standard deviation for particle age (years)'
            plot_name+='_StdAge'
        cb_ticks='years'
    
    if fishing_efforts is True:
        contours_dict = np.load('../Files/contours_dict_sigma1_res02.npy',allow_pickle='TRUE').item()
        poly_areas=[]
        for i, iarea in enumerate(contours_dict['area']):
            if iarea > 5000:
                clippath = Path(np.c_[contours_dict['coords'][i][:,0], contours_dict['coords'][i][:,1]])
                poly_areas.append(clippath)
    else:
        poly_areas=None
                    
    # actual plot
    fig, ax = setmap_subplots(release, vertical = vertical, plot_release=plot_release, 
                              extent=extent, inline_plotting=inline_plotting, 
                              pop_density=pop_density, new_config=new_config, 
                              fishing=fishing_efforts)
    if buffer is False:
        co=[]
        if vertical == 'divided':
            count=0
            for s, (key, Tall_i) in enumerate(Tall.items()):
                for axi in ax.flatten()[count:count+2]:
                    co.append(plt_hist_particles(fig, axi, Tall_i, stype='MERCATOR', 
                                                  unique=unique, log=log, age=age, beaching=beaching,
                                                  method=method, cmap_name=cmap_name))
                count+=2
        else:
            for s, (key, Tall_i) in enumerate(Tall.items()):
                co.append(plt_hist_particles(fig, ax.flatten()[s], Tall_i, stype='MERCATOR', 
                                              unique=unique, hist_depth=hist_depth, log=log, age=age, 
                                              hist2d_windage=hist2d_windage, beaching=beaching, method=method, 
                                              cmap_name=cmap_name, set_clip=poly_areas, imax=imax))
        
        if fishing_efforts is True:
            cmap_gray = copy(plt.get_cmap('gray_r'))
            cmap_gray.set_under('white')     
            co_gray = plt.cm.ScalarMappable(cmap=cmap_gray, 
                                            norm=colors.LogNorm(vmin = 0.1, vmax = 30)) 
            add_colorbar(ax, co[0], plabel, fig=fig, extend=cb_extend, log=log, ticks=cb_ticks,
                          mappable2=co_gray, location='bottom')
            
        else:
            method_max='mean' if unique is True and log is False else 'max'
            set_max(co, imin=imin, imax=imax, method=method_max)
            add_colorbar(ax, co[0], plabel, fig=fig, extend=cb_extend, log=log, ticks=cb_ticks,
                         location='bottom')
    else:
        co=plt_hist_particles_beached_buffer(fig, ax, Tall, var, plot_release, 
                                              log, vertical, plot_name, max_scale=imax)
        add_colorbar(ax, co, plabel, fig=fig, ticks=cb_ticks, location='bottom',
                      extend='max')
    
    # if fishing_efforts is True:
    #     contours_dict = np.load('../Files/contours_dict_sigma1_res02.npy',allow_pickle='TRUE').item()
    #     for i, iarea in enumerate(contours_dict['area']):
    #         if iarea > 5000:
    #             for n, axx in enumerate(ax.flatten()):
    #                 axx.plot(contours_dict['coords'][i][:,0], contours_dict['coords'][i][:,1], 
    #                           'black', linewidth=0.6, linestyle='-', alpha=0.5)
   
    plt.savefig(plot_name, dpi=300) 
    plt.close(fig)

def plot_hist2d_MERCATOR_season(Tall, plot_name, projection=ccrs.PlateCarree(), unique=True):
    import matplotlib.gridspec as gridspec
    
    def get_season(date):
        y=date.year
        seasons = {'Summer':(datetime(y,6,21), datetime(y,9,22)),
                   'Autumn':(datetime(y,9,23), datetime(y,12,20)),
                   'Spring':(datetime(y,3,21), datetime(y,6,20))}
        
        for season,(season_start, season_end) in seasons.items():
            if date>=season_start and date<= season_end:
                return season
        else:
            return 'Winter'

    def get_hist(ax, Tall, index, release_areas, unique=False):
        def count_unique(values):
            return np.unique(values).shape[0]
    
        def deal_with_H(H, xe, ye):
            H[0,: ] = H[-1, :]
            xb = (xe[1:] + xe[:-1])/2
            yb = (ye[1:] + ye[:-1])/2
            xbb, ybb = np.meshgrid(xb, yb)
            xbb[xbb>180]=xbb[xbb>180]-360
            return H, xbb, ybb
        
        cmap = copy(plt.get_cmap("Spectral_r"))
        cmap.extend='max'
        cmap.set_under('white') 
        cmap.set_over('magenta')
        
        nc_lon=np.arange(release_areas['lons'][0], release_areas['lons'][1], 0.25)
        nc_lat=np.arange(release_areas['lats'][0], release_areas['lats'][1], 0.25)
        bins=[nc_lon, nc_lat]
        nc_lon_grid, nc_lat_grid = np.meshgrid(nc_lon, nc_lat)
        non_beached=np.where(Tall.beached[index]==0)
        if unique:
            shape_1=len(np.unique(Tall.trajectory[index]))
            HH, xe, ye,_ = stats.binned_statistic_2d(Tall.lon[index][non_beached], Tall.lat[index][non_beached],
                                                     Tall.trajectory[index][non_beached], bins=bins, 
                                                     statistic=count_unique)
            H=(HH/shape_1)*100
            cnorm=colors.LogNorm(vmin = 0.1, vmax = 100) 
            H, xbb, ybb = deal_with_H(H, xe, ye)
            
            co = ax.pcolormesh(xbb, ybb, H.T, 
                               cmap = cmap,
                               norm=cnorm,
                               zorder = 0)
        else:
            H, xe, ye = np.histogram2d(Tall.lon[index][non_beached], Tall.lat[index][non_beached], bins=bins)
            H, xbb, ybb = deal_with_H(H, xe, ye)
            
            H = np.ma.masked_where(H < np.min(H)+1, H)
            co = ax.pcolormesh(xbb, ybb, H.T, zorder =0, shading='auto', cmap=cmap)
            # co = ax.pcolormesh(xbb, ybb, H.T, zorder =0, shading='gouraud', cmap=cmap)
            # ax.contour(co, colors='k', linewidths=1)
        return co
            
    date_array_seconds=np.arange(np.min(Tall[1].time), np.max(Tall[1].time)+1, 86400)
    date_list_datenum = [Tall[1].time_origin + timedelta(days=x) for x in range(len(date_array_seconds))]
    
    dates_seasons={'0_Winter': [date_array_seconds[index] for index, date in enumerate(date_list_datenum) if get_season(date) == 'Winter'],
                   '1_Spring': [date_array_seconds[index] for index, date in enumerate(date_list_datenum) if get_season(date) == 'Spring'],
                   '2_Summer': [date_array_seconds[index] for index, date in enumerate(date_list_datenum) if get_season(date) == 'Summer'],
                   '3_Autumn': [date_array_seconds[index] for index, date in enumerate(date_list_datenum) if get_season(date) == 'Autumn']}
    
    
    SMALL_SIZE = 11
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 13
    
    release_areas={1: {'name': 'Azores',
                       'lats': [30, 47],
                       'lons': [-38, -21]},
                   2: {'name': 'Madeira',
                       'lats': [30, 46],
                       'lons': [-24.5, -8.5]},
                   3: {'name': 'Canaries',
                       'lats': [27, 45],
                       'lons': [-25, -7]},
                   4: {'name': 'Cape Verde',
                       'lats': [11, 29],
                       'lons': [-28, -10]}}
    
    borders=cfeature.BORDERS
    coast= cfeature.GSHHSFeature(scale='high',facecolor='0.85')
    
    plt.ioff()
    fig = plt.figure(figsize=(9.3, 10), constrained_layout=False)
    outer = gridspec.GridSpec(2, 2, wspace=0.06, hspace=0.06,width_ratios=[3, 3],
                          height_ratios=[3, 3], left=0.01, right=0.99, top=0.99, bottom=0.1)
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fon
    
    outer.tight_layout(fig)
    co=[]
    ax=[]
    for i, arq in enumerate(release_areas.keys()):
        I_area=np.logical_and(np.logical_and((Tall[arq].lon >= release_areas[arq]['lons'][0]), 
                                             (Tall[arq].lon <= release_areas[arq]['lons'][1])),
                              np.logical_and((Tall[arq].lat >= release_areas[arq]['lats'][0]),
                                             (Tall[arq].lat <= release_areas[arq]['lats'][1])))
                
        inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outer[i], wspace=0.18, hspace=0.06)
        
        for j,ss in enumerate(dates_seasons.keys()):
            ax.append(fig.add_subplot(inner[j], projection=projection))
            plt.text(0.05, 0.9 ,ss.split('_')[1],
                     zorder=10000,
                     horizontalalignment='left',
                     verticalalignment='center',
                     transform = ax[-1].transAxes,
                     fontsize=MEDIUM_SIZE,
                     bbox=dict(fc='white',
                               ec='black'))
            
            gl = ax[-1].gridlines(crs=projection, draw_labels=True, linewidth=0.5, color='gray', 
                              alpha=0.5, linestyle='--', zorder=300)
            
            gl.ylabels_right = True if j % 2 ==0 else False
            gl.xlabels_bottom = True if j<2 else False
            gl.xlabels_top = gl.ylabels_left = False
            # if arq == 3:
            #     gl.xlocator = ticker.FixedLocator(range(-28, -0, 5))
            #     gl.ylocator = ticker.FixedLocator(range(27, 45, 3))
            # elif arq == 2:
            #     gl.xlocator = ticker.FixedLocator(range(-24, -0, 3))
            # elif arq == 4:
            #     gl.xlocator = ticker.FixedLocator(range(-28, -0, 3))
            #     gl.ylocator = ticker.FixedLocator(range(11, 29, 3))
            # else:
            
            gl.xlocator = ticker.AutoLocator()
            gl.ylocator = ticker.AutoLocator() 
            lon_formatter = LongitudeFormatter(degree_symbol='')
            lat_formatter = LatitudeFormatter(degree_symbol='')
            gl.xformatter = lon_formatter
            gl.yformatter = lat_formatter

            extent=release_areas[arq]['lons'] + release_areas[arq]['lats']
            ax[-1].set_extent(extent)
            
            ax[-1].add_feature(coast, edgecolor='black', linewidth=0.3, zorder=3)
            ax[-1].add_feature(borders, linewidth=0.3, zorder=3)

            #new method -> first occurrences (initial time) within the season
            # unique_id, unique_id_index=np.unique(Tall[arq].trajectory, return_index=True)
            # I_season_first=np.isin(Tall[arq].time[unique_id_index],dates_seasons[ss])
            # I_season=np.isin(Tall[arq].trajectory, unique_id[I_season_first])
            
            # old method (all particles within the season)
            I_season=np.isin(Tall[arq].time,dates_seasons[ss]) 
            index=np.logical_and(I_season, I_area)
            
            co.append(get_hist(ax[-1], Tall[arq], index, release_areas[arq], unique=unique))
    
    if unique is True:
        imin=0.1
        imax=10.01
        cb_extend='both'
        cb_ticks='log'
        log=True
        plabel='Proportion of particles (%)'
        plot_name+='_hist_seasonal_unique'
    else:
        imin=0
        imax=8000
        cb_extend='max'
        log=False
        cb_ticks=None
        plabel='Number of particle occurrences'
        plot_name+='_hist_seasonal'
            
    set_max(co, imin=imin, imax=imax)
    add_colorbar(ax, co[0], plabel, fig=fig, extend=cb_extend, log=log, ticks=cb_ticks)
    plt.savefig(plot_name, dpi=300) 
    plt.close(fig)
    
def plot_release_boxes(plot_name, projection=ccrs.PlateCarree()):
    release={1: {'name': 'Azores',
                 'lats': [36.5, 40],
                 'lons': [-31.7, -24.8]},
             2: {'name': 'Madeira',
                 'lats': [32.18, 33.55],
                 'lons': [-17.8, -15.8]},
             3: {'name': 'Canaries',
                 'lats': [27.2, 29.7],
                 'lons': [-18.5, -13.2]},
             4: {'name': 'Cape Verde',
                 'lats': [14.5, 17.4],
                 'lons': [-25.6, -22.4]}}
    
    coords={1: {'name': 'Azores_west',
                'lats': [39.25, 39.80],
                'lons': [-31.40, -30.90],
                'start_lon': [],
                'start_lat': []},
            11: {'name': 'Azores_central',
                'lats': [38.25, 39.25],
                'lons': [-29.05, -26.80],
                'start_lon': [],
                'start_lat': []},
            12: {'name': 'Azores_east',
                'lats': [36.80, 38.10],
                'lons': [-26.00, -25.00],
                'start_lon': [],
                'start_lat': []},
            2: {'name': 'Madeira',
                'lats': [32.35, 33.40],
                'lons': [-17.55, -15.95],
                'start_lon': [],
                'start_lat': []},
            3: {'name': 'Canaries_east',
                'lats': [28.00, 29.54],
                'lons': [-14.44, -13.20],
                'start_lon': [],
                'start_lat': []},
            31: {'name': 'Canaries_west',
                'lats': [27.50, 29.02],
                'lons': [-18.25, -15.24],
                'start_lon': [],
                'start_lat': []},
            4: {'name': 'CV_north',
                'lats': [16.45, 17.30],
                'lons': [-25.40, -23.89],
                'start_lon': [],
                'start_lat': []},
            41: {'name': 'CV_east',
                'lats': [15.95, 16.95],
                'lons': [-23.15, -22.60],
                'start_lon': [],
                'start_lat': []},
            42: {'name': 'CV_south',
                'lats': [14.70, 15.45],
                'lons': [-24.90, -22.90],
                'start_lon': [],
                'start_lat': []}}
    
    real_dif_lon=0.08333206
    real_dif_lat=0.083333015
    dif_lon=real_dif_lon*3
    dif_lat=real_dif_lat*3
    
    coords[1]['start_lon'], coords[1]['start_lat']=np.meshgrid(np.arange(coords[1]['lons'][0], coords[1]['lons'][1], dif_lon),
                                                               np.arange(coords[1]['lats'][0], coords[1]['lats'][1], dif_lon))
    coords[11]['start_lon'], coords[11]['start_lat']=np.meshgrid(np.arange(coords[11]['lons'][0], coords[11]['lons'][1], dif_lon),
                                                               np.arange(coords[11]['lats'][0], coords[11]['lats'][1], dif_lon))
    coords[12]['start_lon'], coords[12]['start_lat']=np.meshgrid(np.arange(coords[12]['lons'][0], coords[12]['lons'][1], dif_lon),
                                                               np.arange(coords[12]['lats'][0], coords[12]['lats'][1], dif_lon))
    coords[2]['start_lon'], coords[2]['start_lat']=np.meshgrid(np.arange(coords[2]['lons'][0], coords[2]['lons'][1], dif_lon),
                                                               np.arange(coords[2]['lats'][0], coords[2]['lats'][1], dif_lon))
    coords[3]['start_lon'], coords[3]['start_lat']=np.meshgrid(np.arange(coords[3]['lons'][0], coords[3]['lons'][1], dif_lon),
                                                               np.arange(coords[3]['lats'][0], coords[3]['lats'][1], dif_lon))
    coords[31]['start_lon'], coords[31]['start_lat']=np.meshgrid(np.arange(coords[31]['lons'][0], coords[31]['lons'][1], dif_lon),
                                                               np.arange(coords[31]['lats'][0], coords[31]['lats'][1], dif_lon))
    coords[4]['start_lon'], coords[4]['start_lat']=np.meshgrid(np.arange(coords[4]['lons'][0], coords[4]['lons'][1], dif_lon),
                                                               np.arange(coords[4]['lats'][0], coords[4]['lats'][1], dif_lon))
    coords[41]['start_lon'], coords[41]['start_lat']=np.meshgrid(np.arange(coords[41]['lons'][0], coords[41]['lons'][1], dif_lon),
                                                               np.arange(coords[41]['lats'][0], coords[41]['lats'][1], dif_lon))
    coords[42]['start_lon'], coords[42]['start_lat']=np.meshgrid(np.arange(coords[42]['lons'][0], coords[42]['lons'][1], dif_lon),
                                                               np.arange(coords[42]['lats'][0], coords[42]['lats'][1], dif_lon))
    locs=np.concatenate((np.array([coords[1]['start_lon'].flatten(), coords[1]['start_lat'].flatten(), 
                                   np.repeat(1, len(coords[1]['start_lon'].flatten()))]).T,
                         np.array([coords[11]['start_lon'].flatten(), coords[11]['start_lat'].flatten(), 
                                   np.repeat(1, len(coords[11]['start_lon'].flatten()))]).T,
                         np.array([coords[12]['start_lon'].flatten(), coords[12]['start_lat'].flatten(), 
                                   np.repeat(1, len(coords[12]['start_lon'].flatten()))]).T,
                         np.array([coords[2]['start_lon'].flatten(), coords[2]['start_lat'].flatten(), 
                                   np.repeat(2, len(coords[2]['start_lon'].flatten()))]).T,
                         np.array([coords[3]['start_lon'].flatten(), coords[3]['start_lat'].flatten(), 
                                  np.repeat(3, len(coords[3]['start_lon'].flatten()))]).T,
                         np.array([coords[31]['start_lon'].flatten(), coords[31]['start_lat'].flatten(), 
                                  np.repeat(3, len(coords[31]['start_lon'].flatten()))]).T,
                         np.array([coords[4]['start_lon'].flatten(), coords[4]['start_lat'].flatten(), 
                                   np.repeat(4, len(coords[4]['start_lon'].flatten()))]).T,
                         np.array([coords[41]['start_lon'].flatten(), coords[41]['start_lat'].flatten(), 
                                   np.repeat(4, len(coords[41]['start_lon'].flatten()))]).T,
                         np.array([coords[42]['start_lon'].flatten(), coords[42]['start_lat'].flatten(), 
                                   np.repeat(4, len(coords[42]['start_lon'].flatten()))]).T),
                         axis=0)
    
    coords_boundaries=np.array([locs[:,0]-dif_lon/2, locs[:,0]+dif_lon/2,
                                 locs[:,1]-dif_lat/2, locs[:,1]+dif_lat/2,
                                 locs[:,2]]).T
    
    density_macro=np.array([[0.00, 0.01, -1],
                            [0.01, 0.02, -2],
                            [0.02, 0.03, -3],
                            [0.03, 0.04, -4],
                            [0.04, 0.05, -5]])#MACRO
    row_len_macro=density_macro.shape[0]
    
    len_row=1*len(coords_boundaries)*len(density_macro)
    particles=np.empty((len_row,6),dtype=np.float32)
    interval=row_len_macro*coords_boundaries.shape[0]
    idx=np.concatenate([np.arange(d,interval,coords_boundaries.shape[0]) for d in range(coords_boundaries.shape[0])])
    dens_id = np.concatenate([np.repeat(d, len(coords_boundaries)) for d in density_macro[:,2]])[idx]
    locs_id=np.concatenate([np.repeat(d, density_macro.shape[0]) for d in coords_boundaries[:,4]])
    
    ii=0
    t=1

    random_lon = np.concatenate([(coords_boundaries[d,1] - coords_boundaries[d,0]) * np.random.random_sample(row_len_macro) + coords_boundaries[d,0] for d in range(len(coords_boundaries))])
    random_lat = np.concatenate([(coords_boundaries[d,3] - coords_boundaries[d,2]) * np.random.random_sample(row_len_macro) + coords_boundaries[d,2] for d in range(len(coords_boundaries))])

    dens_macro = np.concatenate([(density_macro[d,1] - density_macro[d,0]) * np.random.random_sample(coords_boundaries.shape[0]) + density_macro[d,0] for d in range(row_len_macro)])[idx]          
    
    particles[ii*interval:(ii+1)*interval, :]=np.vstack([np.repeat(t, interval),
                                                         locs_id,
                                                         random_lon,
                                                         random_lat,
                                                         dens_id,
                                                         dens_macro]).T
        
    # actual plot
    SMALL_SIZE = 12
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15
    
    borders=cfeature.BORDERS
    coast= cfeature.GSHHSFeature(scale='high',facecolor='0.85')
    
    colors_list=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    labels=['[0,1]', '[1,2]', '[2,3]', '[3,4]', '[4,5]']
    
    fig, ax = plt.subplots(figsize=(13, 7), nrows=2, ncols=2,
                           subplot_kw=dict(projection=projection), 
                           constrained_layout=False)
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fon
    
    for n, axx in enumerate(ax.flatten()):
        gl = axx.gridlines(crs=projection, draw_labels=True,
                           linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                           zorder=300)

        gl.ylabels_left = True 
        gl.xlabels_bottom = True 
        gl.xlabels_top = gl.ylabels_right = False

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        # gl.xlocator = ticker.FixedLocator(range(-100, 60, 20))

        axx.set_extent(release[n+1]['lons'] + release[n+1]['lats'])
        axx.add_feature(coast, edgecolor='black', linewidth=0.3, zorder=3)
        axx.add_feature(borders, linewidth=0.3, zorder=3)
    
        for i in range(coords_boundaries.shape[0]):
            lonss=np.array([coords_boundaries[i,0],coords_boundaries[i,1],coords_boundaries[i,1],coords_boundaries[i,0],coords_boundaries[i,0]])
            latss=np.array([coords_boundaries[i,2],coords_boundaries[i,2],coords_boundaries[i,3],coords_boundaries[i,3],coords_boundaries[i,2]])
            axx.plot(lonss,latss,color='black')
        
        for i in range(len(density_macro)):
            p0=density_macro[i, 2]
            ii=np.where(particles[:,4]==p0)
            axx.plot(particles[ii,2][0],particles[ii,3][0],'.', color=colors_list[i], label=labels[i])
            axx.plot(particles[ii,2],particles[ii,3],'.', color=colors_list[i])
        
        if n == 0:
            lg=axx.legend(title='Windage factor (%)', loc=3, fontsize='medium', fancybox=True,
                       markerscale=2)
            lg.zorder=10000
    
    plt.tight_layout(h_pad=3)
    plt.savefig(plot_name+'_release_locations', dpi=300) 
    plt.close(fig)
    
def plot_fishing_and_population(hours=50, sigma=0):
    from scipy.ndimage.filters import gaussian_filter
    # core modules
    from functools import partial
    
    # 3rd pary modules
    import pyproj
    from shapely.geometry import Polygon
    import shapely.ops as ops
    
    def get_contour_verts(cn):
        contours = []
        # for each contour line
        for cc in cn.collections:
            paths = []
            # for each separate section of the contour line
            for pp in cc.get_paths():
                xy = []
                # for each segment of that section
                for vv in pp.iter_segments():
                    xy.append(vv[0])
                paths.append(np.vstack(xy))
            contours.append(paths)
    
        return contours
    
    figsize=(14, 8) 
    borders=cfeature.BORDERS
    coast= cfeature.GSHHSFeature(scale='high')
    facecolor_coast='none'
    extent=[-100, 20, 0, 50]
    border_color='black'
    border_witdh=0.6
    SMALL_SIZE = 12
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 15
        
    prism_extent, prism_array= get_pop_dens()
    X02, Y02, fish_matrix02=get_fishing_hours(res=0.2)
    X01, Y01, fish_matrix01=get_fishing_hours(res=0.1)
    plt.ioff()
    
    #config optima:
    # sigma=2
    # res=0.01
    # min_area = 5000
    # hours= 50
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection=ccrs.PlateCarree()))
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=0.5, color='gray', alpha=0.5, linestyle='--',
                        zorder=300)
    gl.ylabels_left = gl.xlabels_bottom = True 
    gl.xlabels_top = gl.ylabels_right = False 
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_extent(extent)
    ax.add_feature(coast, edgecolor='black', linewidth=0.3, zorder=3, facecolor=facecolor_coast)
    ax.add_feature(borders, linewidth=border_witdh, zorder=3, edgecolor=border_color)
    
    img_plot = ax.imshow(prism_array, 
                            extent=prism_extent,
                            cmap='cmo.gray_r', 
                            norm=colors.LogNorm(vmin=0.1, vmax=10000), 
                            interpolation='nearest', 
                            zorder=2)
    
    # fish=ax.pcolormesh(X01, Y01,fish_matrix01.T, 
    #                     norm=colors.LogNorm(vmin = 0.1, vmax = 1000))

    # data = gaussian_filter(fish_matrix01, sigma=sigma)
    # cn=ax.contour(X01, Y01,data.T, [hours], colors='red', antialiased=True)
    
    # getting areas
    # contours=get_contour_verts(cn)
    # area=np.nan * np.zeros(len(contours[0]))
    # for i, contour_line in enumerate(contours[0]):
    #     poly=Polygon(contour_line)
    #     poly_area = ops.transform(partial(pyproj.transform,
    #                                       pyproj.Proj(init='EPSG:4326'),
    #                                       pyproj.Proj(proj='aea',
    #                                                   lat_1=poly.bounds[1],
    #                                                   lat_2=poly.bounds[3])),
    #                               poly)
    #     area[i]=poly_area.area*10E-7
    
    # Saving contours inpu .npy file 
    # contours_dict={'coords': contours[0],
    #                'area': area,
    #                'sigma': 1,
    #                'res': 0.2}
    # np.save('contours_dict_sigma1_res02.npy', contours_dict) 
    
    contours_dict = np.load('../Files/contours_dict_sigma1_res02.npy',allow_pickle='TRUE').item()

    for i, iarea in enumerate(contours_dict['area']):
        if iarea > 10000:
            ax.plot(contours_dict['coords'][i][:,0], contours_dict['coords'][i][:,1], 'r', linewidth=2)
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fon
    
    cax_left = fig.add_axes([ax.get_position().x0-0.056,
                             ax.get_position().y0,
                             0.02,
                             ax.get_position().y1-ax.get_position().y0]) 
    cb_left=fig.colorbar(img_plot, cax=cax_left, orientation='vertical', 
                                extend='max')
    cb_left.set_label('Population density (persons/km$^{2}$)')
    cax_left.yaxis.set_ticks_position('left')
    cax_left.yaxis.set_label_position('left')
    
    cax_right = fig.add_axes([ax.get_position().x1+0.005,
                              ax.get_position().y0,
                              0.02,
                              ax.get_position().y1-ax.get_position().y0]) 
    cb_right=fig.colorbar(fish, cax=cax_right, orientation='vertical', 
                          extend='max', format='%1d')
    cb_right.set_label('Fishing effort (h/0.1$\degree$)')
    cax_right.yaxis.set_ticks_position('right')
    cax_right.yaxis.set_label_position('right')
    
    cb_right.ax.plot([0,1000],[hours,hours],'r')
    cb_right.ax.plot([0,1000],[150,150],'b')
    
    pre_path='/home/ccardoso/Cenas/Academico/OOM/CleanAtlantic/Papers/Part_1/images'
    try:
        plt.savefig(pre_path+'/final_Fishing_efforts_and_population_res02_h'+str(hours)+'_sigma'+str(sigma), dpi=300) 
    except:
        plt.savefig(pre_path+'/final_Fishing_efforts_and_population_res02_h'+str(hours)+'_sigma'+str(sigma).replace('.',''), dpi=300) 
    
    # plt.savefig(pre_path+'/final_Fishing_efforts_and_population_res02_h'+str(hours)+'_sigma'+str(sigma), dpi=300) 
    
    plt.close(fig)
#%% 
       
def plot(filename, start_date=None, finish_date=None, scatter_age=False, hist2d=False, 
         hist2d_unique=False, hist2d_depth=False, hist2d_age=False, hist2d_windage=False,
         hist_beaching=False, hist_beaching_buffer=False, hist_beaching_buffer_percentage=False, 
         hist_beaching_buffer_age=False, hist2d_season=False, hist2d_fishing=False):

    if socket.gethostname().find('ciimar') != -1:
        pre_path='/media/claudio_nas/parcels_output/MERCATOR/'
        plots_path=pre_path+'../plots/MERCATOR/'
    elif socket.gethostname().find('Latitude') != -1:
        pre_path='/media/claudio_nas/parcels_output/MERCATOR/'
        plots_path=pre_path+'../plots/MERCATOR/'
    else:
        pre_path='/home/otras/oom/mli/LUSTRE/OceanParcels/output/MERCATOR/'
        plots_path=pre_path+'/plots/'+'new_plots/'
        
    organize_ocean = False
    if start_date is not None:
        start_date=datetime.strptime(start_date, '%Y%m%d')
    if finish_date is not None:
        finish_date=datetime.strptime(finish_date, '%Y%m%d')

    if 'kukulka' not in filename.lower():
        varnames=['trajectory', 'lon', 'lat', 'time', 'beached', 'p0']
    elif 'forward' in filename.lower():
        varnames=['trajectory', 'lon', 'lat', 'time']
    else:
        varnames=['trajectory', 'lon', 'lat', 'z', 'time', 'beached'] 
    
        
    release={1: {'name': 'Azores',
                 'lats': [36.80, 39.80],
                 'lons': [-31.40, -25.00]},
             2: {'name': 'Madeira',
                 'lats': [32.35, 33.15],
                 'lons': [-17.55, -16.25]},
             3: {'name': 'Canaries',
                 'lats': [27.50, 29.26],
                 'lons': [-18.25, -13.40]},
             4: {'name': 'Cabo Verde',
                 'lats': [14.70, 17.30],
                 'lons': [-25.40, -22.60]}}
        
    file = pre_path+filename
    plot_name=plots_path+filename[:-3]
    Tall = load_particles_file(file, varnames, start_date, finish_date,
                                organize_ocean=organize_ocean, calculate_age=hist2d_age)
    
    if scatter_age:
        print('Plotting scatter (age)')
        plot_scatter_age_MERCATOR(Tall, release, plot_name)
        
    if hist2d:
        print('Plotting histogram')
        plot_hist2d_MERCATOR(Tall, plot_name, release, log = True)
        
    if hist2d_age:
        #print('Plotting histogram (average age)')
        #plot_hist2d_MERCATOR(Tall, plot_name, release, age=True, method='mean', new_config=True)
        #print('Plotting histogram (median age)')
        #plot_hist2d_MERCATOR(Tall, plot_name+'_5years', release, age=True, method='median', new_config=True)
        print('Plotting histogram (std deviation)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, age=True, method='std', new_config=True)
        
    if hist2d_unique:
        print('Plotting histogram (unique values i.e. proportion)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, unique=True, log=True, new_config=True)
    
    if hist2d_fishing:
        print('Plotting histogram with fishing efforts (unique values i.e. proportion)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, unique=True, log=True, fishing_efforts=True,
                             new_config=True)
        
    if hist2d_depth:
        # print('Plotting histogram (mean depth)')
        # plot_hist2d_MERCATOR(Tall, plot_name, release, hist_depth=True, method='mean')
        # print('Plotting histogram (median depth)')
        # plot_hist2d_MERCATOR(Tall, plot_name, release, hist_depth=True, method='median')
        print('Plotting histogram (max depth)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, hist_depth=True, method='max')
        # print('Plotting histogram (std dev depth)')
        # plot_hist2d_MERCATOR(Tall, plot_name, release, hist_depth=True, method='std')
    
    if hist2d_windage:
        # print('Plotting histogram (mean windage)')
        # plot_hist2d_MERCATOR(Tall, plot_name, release, hist2d_windage=True, method='mean')
        # print('Plotting histogram (mode windage)')
        # plot_hist2d_MERCATOR(Tall, plot_name, release, hist2d_windage=True, method='mode')
        # print('Plotting histogram (mode windage)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, hist2d_windage=True, method='mode_unique',
                             new_config=True)
        print('Plotting histogram (mode windage) UNIQUE!')
        
    if hist_beaching:
        print('Plotting histogram for beached particles (pcolormesh)')
        plot_hist2d_MERCATOR(Tall, plot_name, release, beaching=True, buffer=False,
                             log=True)
    if hist_beaching_buffer:
        print('Plotting buffers along coastlines for beached particles')
        plot_hist2d_MERCATOR(Tall, plot_name, release, beaching=True, buffer=True)
        
    if hist_beaching_buffer_percentage:
        if 'backward' in file:
            imax=40
        else:
            imax=15
        print('Plotting buffers (with percentage) along coastlines for beached particles')
        plot_hist2d_MERCATOR(Tall, plot_name, release, beaching=True, buffer=True, percentage=True,
                             pop_density=True, imax=imax)
        
    if hist_beaching_buffer_age:
        print('Plotting buffers (with age) along coastlines for beached particles')
        plot_hist2d_MERCATOR(Tall, plot_name, release, beaching=True, buffer=True, age = True,
                             method='mean')
        
    if hist2d_season:
        print('Plotting seasonal histogram')
        plot_hist2d_MERCATOR_season(Tall, plot_name)

        
if __name__=="__main__":
    p = ArgumentParser(description="""Macaronesia case study""")
    p.add_argument('-filename', '--filename', default='MACARONESIA_backward_v00.nc', help='filename to plot')
    p.add_argument('-start_date', '--start_date', default=None, help='Date from which particles were released (YYYYMMDD)')
    p.add_argument('-finish_date', '--finish_date', default=None, help='Date until when particles were released (YYYYMMDD)')
    p.add_argument('-scatter_age', '--scatter_age', type=bool, default=False, help='plotting scatter_age')
    p.add_argument('-hist2d', '--hist2d', type=bool, default=False, help='plotting 2D histogram')
    p.add_argument('-hist2d_unique', '--hist2d_unique', type=bool, default=False,help='plotting a 2D histogram for unique values')
    p.add_argument('-hist2d_fishing', '--hist2d_fishing', type=bool, default=False,help='plotting a 2D histogram for unique values')
    p.add_argument('-hist2d_depth', '--hist2d_depth', type=bool, default=False,help='plotting a 2D histogram for particle mean depth')
    p.add_argument('-hist2d_age', '--hist2d_age', type=bool, default=False,help='plotting 2D histogram (average age)')
    p.add_argument('-hist2d_windage', '--hist2d_windage', type=bool, default=False,help='plotting 2D histogram (windage)')
    p.add_argument('-hist_beaching', '--hist_beaching', type=bool, default=False,help='plotting histogram for beached particles only')
    p.add_argument('-hist_beaching_buffer', '--hist_beaching_buffer', type=bool, default=False,help='Plotting buffers along coastlines for beached particles')
    p.add_argument('-hist_beaching_buffer_percentage', '--hist_beaching_buffer_percentage', type=bool, default=False,help='Plotting buffers along coastlines for beached particles')
    p.add_argument('-hist_beaching_buffer_age', '--hist_beaching_buffer_age', type=bool, default=False,help='Plotting buffers along coastlines for beached particles')
    p.add_argument('-hist2d_season', '--hist2d_season', type=bool, default=False, help='plot seasonal histogram')
    args = p.parse_args()
    print('These are the arguments: ' + str(args))
    
    plot(filename=args.filename, finish_date=args.finish_date, start_date=args.start_date, 
          scatter_age=args.scatter_age, 
          hist2d=args.hist2d, 
          hist2d_unique=args.hist2d_unique, 
          hist2d_fishing=args.hist2d_fishing,
          hist2d_depth=args.hist2d_depth, 
          hist2d_age=args.hist2d_age, 
          hist2d_windage=args.hist2d_windage,
          hist_beaching=args.hist_beaching, 
          hist_beaching_buffer=args.hist_beaching_buffer, 
          hist_beaching_buffer_percentage=args.hist_beaching_buffer_percentage, 
          hist_beaching_buffer_age=args.hist_beaching_buffer_age, 
          hist2d_season=args.hist2d_season)
