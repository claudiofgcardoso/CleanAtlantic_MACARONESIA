#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:12:21 2020

@author: ccardoso
"""
import random
import math
from glob import glob
from parcels import (FieldSet, VectorField, Field, NestedField, ParticleSet, 
                     Variable, JITParticle, ErrorCode)
import numpy as np
from operator import attrgetter
import time
import os
print("Current working directory is:", os.getcwd() ) 
import socket 
from argparse import ArgumentParser
from datetime import datetime, timedelta


#%% utils
def get_MERCATOR(path, start_date=0, finish_date=-1, chunksize=False, fieldset=None):
    fdir=sorted(glob(path + '*.nc'))
    len_path=len(path)   
    
    if type(start_date) == datetime:
        start_date_str=start_date.strftime("%Y%m%d")
        finish_date_str=finish_date.strftime("%Y%m%d")
        for i, e in enumerate(fdir):
            if e.find(str(start_date_str),len_path) > 0:
                start_index = i 
            elif e.find(str(finish_date_str),len_path) > 0:
                finish_index = i 
             
    if start_date > finish_date:
        filenames = {'U': fdir[finish_index:start_index+1],
                     'V': fdir[finish_index:start_index+1]}
    else:
        filenames = {'U': fdir[start_index:finish_index+1],
                     'V': fdir[start_index:finish_index+1]}  
    
    if path.find('0083') > 0:
        variables = {'U': 'uo',
                     'V': 'vo'}
    else:
        variables = {'U': 'u',
                     'V': 'v'}    
            
    dimensions = {'U': {'time': 'time', 'depth': 'depth', 'lon': 'longitude', 'lat': 'latitude'},
                  'V': {'time': 'time', 'depth': 'depth', 'lon': 'longitude', 'lat': 'latitude'}}
        
    if fieldset:
        fieldset.Unemo_old = fieldset.U
        fieldset.Unemo_old.name = 'Unemo_old'
        fieldset.Vnemo_old = fieldset.V
        fieldset.Vnemo_old.name = 'Vnemo_old'
        
        fieldset_new = FieldSet.from_nemo(filenames, variables, dimensions, vmax = 5, vmin= -5, 
                                          field_chunksize=chunksize)
        fieldset_new.U.name='Unemo_new'
        fieldset_new.V.name='Vnemo_new'
        
        fieldset.add_field(fieldset_new.U)
        fieldset.add_field(fieldset_new.V)
        
        fieldset.U = NestedField('U', [fieldset.Unemo_old, fieldset.Unemo_new])
        fieldset.V = NestedField('V', [fieldset.Vnemo_old, fieldset.Vnemo_new])

        
    else:
        fieldset = FieldSet.from_nemo(filenames, variables, dimensions, vmax = 5, vmin= -5, 
                                      field_chunksize=chunksize)
    return fieldset

def get_stokes_MERCATOR(fieldset, path, start_date=0, finish_date=-1, chunksize=False):
    fdir=sorted(glob(path + '*.nc'))
    len_path=len(path)
    
    if type(start_date) == datetime:
        start_date=start_date.strftime("%Y%m%d")
        finish_date=finish_date.strftime("%Y%m%d")
        for i, e in enumerate(fdir):
            if e.find(str(start_date),len_path) > 0:
                start_index = i 
            elif e.find(str(finish_date),len_path) > 0:
                finish_index = i 

    if start_date > finish_date:
        filenames = {'U': fdir[finish_index:start_index+1],
                     'V': fdir[finish_index:start_index+1]}
    else:
        filenames = {'U': fdir[start_index:finish_index+1],
                     'V': fdir[start_index:finish_index+1]}
    variables = {'U': 'uuss',
                 'V': 'vuss'}
    dimensions = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}
    Ustokes = Field.from_netcdf(filenames['U'], variables['U'], dimensions, fieldtype='U', 
                                vmax = 2, vmin=-2, allow_time_extrapolation=False, 
                                field_chunksize=chunksize)
    Vstokes = Field.from_netcdf(filenames['V'], variables['V'], dimensions, fieldtype='V',
                             vmax = 2, vmin=-2, grid=Ustokes.grid, dataFiles=Ustokes.dataFiles, 
                             allow_time_extrapolation=False, field_chunksize=chunksize)
    fieldset.add_field(Ustokes, 'uuss')
    fieldset.add_field(Vstokes, 'vuss')
    UV_Stokes = VectorField('UV_Stokes', fieldset.uuss, fieldset.vuss)
    fieldset.add_vector_field(UV_Stokes)
    return fieldset


def get_wind_MERCATOR(fieldset, path, start_date=0, finish_date=-1, chunksize=False):
    fdir=sorted(glob(path + '*.nc'))
    len_path=len(path)
    
    if type(start_date) == datetime:
        start_date=start_date.strftime("%Y%m%d")
        finish_date=finish_date.strftime("%Y%m%d")
        for i, e in enumerate(fdir):
            if e.find(str(start_date),len_path) > 0:
                start_index = i 
            elif e.find(str(finish_date),len_path) > 0:
                finish_index = i 
    
    if start_date > finish_date:
        filenames = {'uwnd': fdir[finish_index:start_index+1],
                     'vwnd': fdir[finish_index:start_index+1]}
    else:
        filenames = {'uwnd': fdir[start_index:finish_index+1],
                     'vwnd': fdir[start_index:finish_index+1]}
    variables = {'uwnd': 'uwnd',
                 'vwnd': 'vwnd'}
    dimensions = {'time': 'time',
                  'lat': 'latitude',
                  'lon': 'longitude'}
    Uwnd = Field.from_netcdf(filenames['uwnd'], variables['uwnd'], dimensions, fieldtype='U', 
                                vmax = 150, vmin=-150, allow_time_extrapolation=False,
                                field_chunksize=chunksize)
    Vwnd = Field.from_netcdf(filenames['vwnd'], variables['vwnd'], dimensions, fieldtype='V',
                             vmax = 150, vmin=-150, grid=Uwnd.grid, dataFiles=Uwnd.dataFiles, 
                             allow_time_extrapolation=False, field_chunksize=chunksize)
    fieldset.add_field(Uwnd, 'uwnd')
    fieldset.add_field(Vwnd, 'vwnd')
    UV_wnd = VectorField('UV_wnd', fieldset.uwnd, fieldset.vwnd)
    fieldset.add_vector_field(UV_wnd)
    return fieldset

def set_diffusion_MERCATOR(fieldset, diffusivity): # create fields needed for the brownian motion
    if type(fieldset.U)==Field:
        size2D = (fieldset.U.grid.ydim, fieldset.U.grid.xdim)
        lat1=fieldset.U.grid.lat
        lon1=fieldset.U.grid.lon
        fieldset.add_field(Field('Kh_zonal', data=diffusivity*np.ones(size2D),
                             lon=lon1, lat=lat1,
                             allow_time_extrapolation=True))
        fieldset.add_field(Field('Kh_meridional', data=diffusivity*np.ones(size2D),
                             lon=lon1, lat=lat1,
                             allow_time_extrapolation=True))
    else:
        size2D_0083 = (fieldset.U[0].grid.ydim, fieldset.U[0].grid.xdim)  
        size2D_025 = (fieldset.U[1].grid.ydim, fieldset.U[1].grid.xdim) 
        Kh_zonal = NestedField('Kh_zonal', 
                               [Field('Kh_zonal', data=diffusivity*np.ones(size2D_0083), 
                                      lat=fieldset.U[0].grid.lat,
                                      lon=fieldset.U[0].grid.lon, mesh='spherical', 
                                      allow_time_extrapolation=True),
                                Field('Kh_zonal', data=diffusivity*np.ones(size2D_025), 
                                      lat=fieldset.U[1].grid.lat, 
                                      lon=fieldset.U[1].grid.lon, mesh='spherical', 
                                      allow_time_extrapolation=True)])
        Kh_meridional = NestedField('Kh_meridional', 
                                    [Field('Kh_meridional', data=diffusivity*np.ones(size2D_0083), 
                                           lat=fieldset.U[0].grid.lat,
                                           lon=fieldset.U[0].grid.lon, mesh='spherical', 
                                           allow_time_extrapolation=True),
                                     Field('Kh_meridional', data=diffusivity*np.ones(size2D_025), 
                                           lat=fieldset.U[1].grid.lat,
                                           lon=fieldset.U[1].grid.lon, mesh='spherical', 
                                           allow_time_extrapolation=True)])
        fieldset.add_field(Kh_zonal)
        fieldset.add_field(Kh_meridional)
    return fieldset


def MERCATOR_get_particles_macaronesia(fieldset, start_date=0, stop_release=None, finish_date=None, 
                                       timestep=3600*24, plot=False):
    #deal with timesteps first
    try:
        time_origin=datetime.strptime(str(fieldset.U[0].grid.time_origin)[0:10]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
    except:
        time_origin=datetime.strptime(str(fieldset.U.grid.time_origin)[0:10]+' 00:00:00', '%Y-%m-%d %H:%M:%S')
    
    if finish_date is None:
        finish_date=time_origin+timedelta(seconds=fieldset.U[0].grid.time[-1])
    
    if start_date != 0:
        start_date_h=(start_date-time_origin).total_seconds()
    
    if stop_release is not None and stop_release != finish_date:
        stop_release_h=(stop_release-time_origin).total_seconds()
    else:
        stop_release_h=fieldset.U[0].grid.time[-1]
    
    if start_date > stop_release:
        timesteps=np.arange(start_date_h, stop_release_h+1, timestep*-1)
    else:
        timesteps=np.arange(start_date_h, stop_release_h+1, timestep)
     
    # define coordinates and density/radius classes
    try:
        lon=fieldset.U[0].lon
        lat=fieldset.U[0].lat
    except:
        lon=fieldset.U.lon
        lat=fieldset.U.lat
    
    real_dif_lon=(lon[1]-lon[0])
    real_dif_lat=(lat[1]-lat[0])
    dif_lon=real_dif_lon*3
    dif_lat=real_dif_lat*3
    
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
    
    if plot:
        import matplotlib.pyplot as plt
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs
        try:
            lon=fieldset.U[0].lon
            lat=fieldset.U[0].lat
        except:
            lon=fieldset.U.lon
            lat=fieldset.U.lat
                
        fieldset.computeTimeChunk(3600,3600)
        try:
            U_field = np.ma.getdata(fieldset.U[0].data[0,:][0]) 
        except:
            U_field = np.ma.getdata(fieldset.U.data[0,:][0]) 
        U_field = np.ma.masked_where(U_field == 0, U_field)
        
        extent=[np.min(lon), np.max(lon), np.min(lat), np.max(lat)]

        fig, ax = plt.subplots(figsize=(16,10), subplot_kw=dict(projection=ccrs.PlateCarree()))
        
        ax.set_extent(extent)
        coast= cfeature.GSHHSFeature(scale='high', facecolor='0.85',zorder=3)
        ax.add_feature(coast, edgecolor='black', linewidth=0.4, zorder=3)
            
        # plt.pcolor(lon, lat, U_field)
        plt.plot(locs[:,0],locs[:,1], '.', color='red')
        
        for i in range(coords_boundaries.shape[0]):
            lonss=np.array([coords_boundaries[i,0],coords_boundaries[i,1],coords_boundaries[i,1],coords_boundaries[i,0],coords_boundaries[i,0]])
            latss=np.array([coords_boundaries[i,2],coords_boundaries[i,2],coords_boundaries[i,3],coords_boundaries[i,3],coords_boundaries[i,2]])
            ax.plot(lonss,latss,color='black')

        # ii=np.where(particles[:,1]==2)
        # ax.plot(particles[ii,2],particles[ii,3],'.',color='grey')
            
    len_row=len(timesteps)*len(coords_boundaries)*len(density_macro)
    particles=np.empty((len_row,6),dtype=np.float32)
    interval=row_len_macro*coords_boundaries.shape[0]
    idx=np.concatenate([np.arange(d,interval,coords_boundaries.shape[0]) for d in range(coords_boundaries.shape[0])])
    dens_id = np.concatenate([np.repeat(d, len(coords_boundaries)) for d in density_macro[:,2]])[idx]
    locs_id=np.concatenate([np.repeat(d, density_macro.shape[0]) for d in coords_boundaries[:,4]])
    
    for ii, t in enumerate(timesteps):
        print('Creating ocean particles --> ' + str(round((ii/len(timesteps))*100,2))+'% ')
        
        random_lon = np.concatenate([(coords_boundaries[d,1] - coords_boundaries[d,0]) * np.random.random_sample(row_len_macro) + coords_boundaries[d,0] for d in range(len(coords_boundaries))])
        random_lat = np.concatenate([(coords_boundaries[d,3] - coords_boundaries[d,2]) * np.random.random_sample(row_len_macro) + coords_boundaries[d,2] for d in range(len(coords_boundaries))])
    
        dens_macro = np.concatenate([(density_macro[d,1] - density_macro[d,0]) * np.random.random_sample(coords_boundaries.shape[0]) + density_macro[d,0] for d in range(row_len_macro)])[idx]          
        
        particles[ii*interval:(ii+1)*interval, :]=np.vstack([np.repeat(t, interval),
                                                             locs_id,
                                                             random_lon,
                                                             random_lat,
                                                             dens_id,
                                                             dens_macro]).T

    return particles


def CheckParticlesOnLand_MERCATOR(fieldset, locs): 
    """
    Function to creat a Field with 1's at land points and 0's at ocean points and then
    check if initial position of the particles are located on land or at sea. If on land,
    particles will be deleted.
    """  
    fieldset.computeTimeChunk(0,1)
    f_u=fieldset.U[0].data[0,:]
    f_v=fieldset.V[0].data[0,:]
    f_uv=np.sqrt(f_u**2+f_v**2)
    a_u = np.ma.masked_equal(f_uv, 0)
    aa_u = np.ma.masked_invalid(a_u)
    fieldset.add_field(Field('Land_UV', aa_u.mask*1, 
                             lat = fieldset.U[0].grid.lat,
                             lon=fieldset.U[0].grid.lon,
                             depth=fieldset.U[0].grid.depth,
                             mesh='spherical',
                             interp_method='nearest'))
    
    jj=[]
    for i in range(locs.shape[0]):
        lu = fieldset.Land_UV[0, fieldset.Land_UV.grid.depth[0], locs[i,3], locs[i,2]]
        if lu > 0:
            jj.append(i)
            print("Particle: %d from area: %d initiated on Land. Deleting" % 
                  (i, locs[i,1]))
    if jj:
        locs=np.delete(locs, np.asarray(jj), axis=0) 
        
    return locs

#%% parcels kernels
def DeleteParticle(particle, fieldset, time):
    print("Particle lost!! ID: %g --> (lon: %g, lat: %g, Depth: %g, Time: %g)" % 
      (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()
    
def SubmergeParticle(particle, fieldset, time):
    print('Particle [%d] moved outside depth domain -> %g m. Activating recovery kernel...' % 
          (particle.id, particle.depth))
    particle.depth = 0 
    particle.surface = 1
    
def SubmergeParticle_PASSIVE(particle, fieldset, time):
    print('Particle [%d] moved outside depth domain -> %g m. Activating recovery kernel...' % 
          (particle.id, particle.depth))
    particle.depth = 0 
    particle.time = time + particle.dt
    
def BeachTesting_MERCATOR(particle, fieldset, time):
    """
    Check if particle is beached and attribute following behaviour: 
        0 at sea
        1 beached (delete particle)
        2 after non-beach dynamic
        3 after beach dynamic
        4 unbeach conditionally (tide and beach Ts)
        5 unbeach unconditionally  
    """
    if particle.beached == 2 or particle.beached == 3:
        if fieldset.track_kernels: print('Particle [%d] check beaching' %particle.id)
        (U, V) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if math.fabs(U) < 1e-14 and math.fabs(V) < 1e-14:
            if fieldset.unbeach == False:
                # print("Particle [%d] deleting (beached depth = %g m)!" % 
                #       (particle.id, particle.depth))
                particle.beached = 1
                particle.delete()
            else:
                particle.beached = 5
        else:
            particle.beached = 0 

def UnBeaching_PrevCoord(particle, fieldset, time):
    if particle.beached == 5:
        if fieldset.track_kernels: print('Particle [%d] UNBEACHING' %particle.id)
        particle.lon = particle.prev_lon 
        particle.lat = particle.prev_lat
        particle.depth = particle.prev_depth
        particle.beached = 0
        # particle.unbeachCount += 1
#%% advection kernels

def AdvectionRK4_MERCATOR(particle, fieldset, time):
    """
    A simple advection kernel with current and wind combined
    All field velocities are retrieved at the same location. 
    Final position is calculated at the end of the kernel
    """
    if particle.beached == 0:
        if fieldset.track_kernels: print('Particle [%d] advection' %particle.id)
        # current
        particle.prev_lon = particle.lon  # Set the stored values for next iteration.
        particle.prev_lat = particle.lat   
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        Uc=(u1 + 2*u2 + 2*u3 + u4) / 6.
        Vc=(v1 + 2*v2 + 2*v3 + v4) / 6.
        # wind and stokes
        if particle.depth <= 0.52:
            (Uw, Vw) = fieldset.UV_wnd[time, 0, particle.lat, particle.lon]
            (Ustokes, Vstokes) = fieldset.UV_Stokes[time, 0, particle.lat, particle.lon]
        elif particle.depth > 0.52:
            Uw = 0 
            Vw = 0 
            Ustokes = 0 
            Vstokes = 0
 
        particle.lon += (Uc + (particle.p0 * Uw) + Ustokes) * particle.dt
        particle.lat += (Vc + (particle.p0 * Vw) + Vstokes) * particle.dt
        particle.beached == 2

def kukulka_mixing_MERCATOR(particle, fieldset, time):  
    """
    From Wichmann et al., 2019
    :Kernel that randomly distributes particles along the vertical according to an expovariate distribution.
    :Parameterization according to Kukulka et al. (2012): The effect of wind mixing on the vertical distribution of buoyant plastic debris
    :Comment on dimensions: tau needs to be in Pa
    """
    if particle.beached == 0:
        if fieldset.track_kernels: print('Particle [%d] kukulka' %particle.id) 
        roh=1.2; # kg/m^3, air density
        (U_wnd0, V_wnd0) = fieldset.UV_wnd[time, 0, particle.lat, particle.lon]
        U_wnd=U_wnd0*1852*60*math.cos(particle.lat*math.pi/180)
        V_wnd=V_wnd0*1852*60
        U=math.sqrt(U_wnd**2+V_wnd**2) # Wind speed
        if U <= 1:
            Cd=0.00218
        elif U > 1 or U <= 3:
            Cd=(0.62+1.56/U)*0.001
        elif U > 3 or U < 10:
            Cd=0.00114
        else:
            Cd=(0.49+0.065*U)*0.001
        Tau=Cd*roh*U**2  # kg/m^3*m/s*m/s= N/m^2 = Pa
        A0=0.31 * math.pow(Tau,1.5)
        l=fieldset.wrise/A0
        d=random.expovariate(l) #Kukulka formula. Used depths of [0 ... 108.] m
        if d>108:
            particle.depth=108
        elif d < 0.51:
            particle.depth=0.51       
        else:
            particle.depth=d 

def DiffusionUniformKh(particle, fieldset, time):
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.
    Assumes that fieldset has fields Kh_zonal and Kh_meridional.
    This kernel neglects gradients in the diffusivity field and is
    therefore more efficient in cases with uniform diffusivity.
    Since the perturbation due to diffusion is in this case spatially
    independent, this kernel contains no advection and can be used in
    combination with a seperate advection kernel.
    The Wiener increment `dW` should be normally distributed with zero
    mean and a standard deviation of sqrt(dt). Instead, here a uniform
    distribution with the same mean and std is used for efficiency and
    to keep random increments bounded. This substitution is valid for
    the convergence of particle distributions. If convergence of
    individual particle paths is required, use normally distributed
    random increments instead. See Gräwe et al (2012)
    doi.org/10.1007/s10236-012-0523-y for more information.
    """
    if particle.beached == 0:
        if fieldset.track_kernels: print('Particle [%d] difusion' %particle.id) 
        # particle.prev_lon = particle.lon  # Set the stored values for next iteration.
        # particle.prev_lat = particle.lat   
        # Wiener increment with zero mean and std of sqrt(dt)
        dWx = random.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
        dWy = random.uniform(-1., 1.) * math.sqrt(math.fabs(particle.dt) * 3)
    
        bx = math.sqrt(2 * fieldset.Kh_zonal[0, 0, particle.lat, particle.lon])
        by = math.sqrt(2 * fieldset.Kh_meridional[0, 0, particle.lat, particle.lon])
    
        particle.lon += bx * dWx
        particle.lat += by * dWy
        particle.beached = 3
#%% setting up particles and simulation

def create_pset_MERCATOR(fieldset, particles, calculate_distance=False, ageing=False):
    """

    function to create pset object for the release of particles from the ocean 
    and rom rivers with ROMS currents

    """
    class PlasticParticle(JITParticle):  #JITParticle or ScipyParticle
        loc = Variable('loc', dtype=np.int16, initial=0, to_write="once") #loc = 0 for oceanic origin
        beached = Variable('beached', dtype=np.int8, initial=0)
        micro=Variable('micro', dtype=np.int8, initial=0, to_write="once")
        unbeachCount = Variable('unbeachCount', dtype=np.int16, initial=0, to_write=False)
        prev_lon = Variable('prev_lon', dtype=np.float32,
                        initial=attrgetter('lon'),
                        to_write=False)  # the previous longitude
        prev_lat = Variable('prev_lat', dtype=np.float32, 
                        initial=attrgetter('lat'), 
                        to_write=False)  # the previous latitude.
        p0=Variable('p0', dtype=np.float32, initial=0, to_write="once") # particle density in g/m3
        # Ws=Variable('Ws', dtype=np.float32, initial=0, to_write=False) # Settling velocity (m/s)
        surface=Variable('surface', dtype=np.int8, initial=0, to_write=False) # flag to activate surface advection (and windage)
        if calculate_distance:
            distance = Variable('distance', initial=0., dtype=np.float32)
        if ageing:
            age = Variable('age', dtype=np.int32, initial=0)
    
    # seeding_depths = np.array([0.1, .5, 1.])
    seeding_depths = np.array([0.52])
    depths = np.repeat(seeding_depths, len(particles))
    particles=np.tile(particles, (len(seeding_depths),1))
    pset = ParticleSet(fieldset, PlasticParticle,
                       time=particles[:, 0],
                       loc=particles[:,1],
                       lon=particles[:, 2],
                       lat=particles[:, 3],
                       micro = particles[:,4],
                       p0=particles[:, 5],
                       depth = depths,
                       lonlatdepth_dtype=np.float32)

    return pset


def execute_run(pset, kernel, fieldset, start_date, finish_date, output_file, args,
                timestep=None, stype='ROMS', passive=False):
    
    if start_date > finish_date:
        dt=-timedelta(seconds=fieldset.U[0].grid.time[1]) if timestep is None else -timestep
    else:
        dt=timedelta(seconds=fieldset.U[0].grid.time[1]) if timestep is None else timestep
    runtime=abs(finish_date-start_date) if start_date != finish_date else timedelta(1)-timedelta(hours=1)    # the total length of the run
    tic = time.time()  
    
    print('Running only one simulation: from ' + 
          start_date.strftime("%Y/%m/%d") + " to " + finish_date.strftime("%Y/%m/%d"))
    
    if stype == 'ROMS' and passive is False:
        dict_recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                       ErrorCode.ErrorThroughSurface: SubmergeParticle}
    elif passive is True:
        dict_recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle,
                       ErrorCode.ErrorThroughSurface: SubmergeParticle_PASSIVE}
    else:
        dict_recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}
    
    pset.execute(kernel,                 
                 runtime=runtime,
                 dt=dt,
                 output_file=output_file,
                 recovery=dict_recovery)
    
    #adding metadata to file
    output_file.add_metadata('start_date', start_date.strftime('%Y-%m-%d'))
    output_file.add_metadata('finish_date', finish_date.strftime('%Y-%m-%d'))
    for arg in vars(args):
        output_file.add_metadata(arg, str(getattr(args, arg)))
    output_file.export()
    time_of_execution=str(timedelta(seconds=time.time()-tic))
    print('FINISHED! Execution of simulation took ' + time_of_execution)
    

#%% Actual RUN!
def p_advect(name, stype, unbeach, kukulka, kh, wind, stokes, dt, biorate, biodens,
             chunksize, track_kernels, arguments):

    if socket.gethostname().find('ciimar-madeira') != -1 or socket.gethostname().find('Latitude') != -1:
        pre_path='/media/claudio_nas/MERCATOR_data/'
        path_nemo_0083 = pre_path + 'MERCATOR_0083res/'
        path_nemo_025 = pre_path + 'MERCATOR_025res/'
        path_stokes = pre_path + 'WW3_ECMWF/'
        path_wind = pre_path + 'WIND_ECMWF/'
        outfile='/media/claudio_nas/parcels_output/MERCATOR/'
    else:
        raise ValueError('Script is not prepared to run on this computer!')
    
    outfile+=name
    
    if stype=='backward':
        start_date=datetime(2015,12,29)
        stop_release=datetime(2009,1,1)
        finish_date=datetime(2006,1,1)
        outfile +='_backward'
    elif stype=='forward':
        start_date=datetime(2006,1,1)
        stop_release=datetime(2013,1,1)
        finish_date=datetime(2015,12,29)   
        outfile += '_forward'
    elif stype=='test':
        print(stype)
        start_date=datetime(2006,1,30)
        stop_release=datetime(2006,1,25)
        finish_date=datetime(2006,1,1)
        outfile += '_TESTE'

    if kukulka:
        outfile+='_kukulka'
        
    timestep = dt * 24 * 3600
    if chunksize not in ['auto', 'False', False]:
        chunksize=int(chunksize)
    elif chunksize in ['False']:
        chunksize=False

    # setting fieldsets from NEMO: 0.083 and 0.25 resolution in nested grids 
    if path_nemo_0083:
        fieldset = get_MERCATOR(path_nemo_0083, start_date, finish_date, chunksize=chunksize) 
    if path_nemo_025:
        fieldset = get_MERCATOR(path_nemo_025, start_date, finish_date, chunksize=chunksize, fieldset=fieldset) 
    fieldset.track_kernels=track_kernels
    
    if stokes:
        fieldset = get_stokes_MERCATOR(fieldset, path_stokes, start_date, finish_date, chunksize=chunksize)
    if wind:
        fieldset = get_wind_MERCATOR(fieldset, path_wind, start_date, finish_date, chunksize=chunksize)

    #locations for the release of particles_________________________________________________________________________________________________
    locs=MERCATOR_get_particles_macaronesia(fieldset, start_date, stop_release, finish_date, timestep)
    locs=CheckParticlesOnLand_MERCATOR(fieldset, locs)
    
    #setting constants _____________________________________________________________________________________________________________________
    fieldset.add_constant('track_kernels',track_kernels)
    fieldset.add_constant('pair',1.2E-3) #g/cm3
    fieldset.add_constant('wrise',0.001) #m/s
    fieldset.add_constant('unbeach', unbeach) # define what to do if particle is beached (False = stay there and delete, True = unbeach)
    
    # Load other fields to fieldset ________________________________________________________________________________________________________
    if kh > 0:
        fieldset=set_diffusion_MERCATOR(fieldset, kh)
        
    # Setting particle class ______________________________________________________________________________________________________________
    pset = create_pset_MERCATOR(fieldset, locs)

    # Settinng Kernels_____________________________________________________________________________________________________________________
    kernel = pset.Kernel(AdvectionRK4_MERCATOR) + pset.Kernel(BeachTesting_MERCATOR)
    
    if kukulka:
        kernel += pset.Kernel(kukulka_mixing_MERCATOR) 
    if kh > 0: 
        kernel += pset.Kernel(DiffusionUniformKh) 
    
    kernel += pset.Kernel(BeachTesting_MERCATOR)
    # Now the unbeaching (or not) of particles 
    if unbeach:    
        kernel += pset.Kernel(UnBeaching_PrevCoord) 
    
    while os.path.isfile(outfile):
        print('Output file exists! Adding version to file name')
        path, file = os.path.split(outfile)
        fversion=file[-5:-3]
        nversion="{:02d}".format(int(fversion)+1) + '.nc'
        outfile=path + '/' + file[0:-5] + nversion
        
    output_file = pset.ParticleFile(name=outfile, outputdt=timedelta(days=1))       
    # executing simulation
    execute_run(pset, kernel, fieldset, start_date, finish_date, output_file, 
                args, stype='MERCATOR')
    
    
if __name__=="__main__":
    p = ArgumentParser(description="""Madeira case study - Regional scenario""")
    p.add_argument('-name', '--name', type=str, default='MACARONESIA',help='name of the run')
    p.add_argument('-stype', '--stype',choices=('test', 'forward', 'backward'), nargs='?', default='test',help='execution mode')
    p.add_argument('-unbeach', '--unbeach', type=bool, default=False,help='unbeach partiles after stokes or wind interaction')
    p.add_argument('-kukulka', '--kukulka', type=bool, default=False,help='kukulka vertical mixing')
    p.add_argument('-kh', '--kh', type=int, default=10,help='diffusion Kh in m^2/s')
    p.add_argument('-wind', '--wind', type=bool, default=True,help='include windage effects on particles')
    p.add_argument('-stokes', '--stokes', type=bool, default=True,help='include stokes drift on particles')
    p.add_argument('-dt', '--dt', type=int, default=3,help='dt for the release of particles (daily)')
    p.add_argument('-biorate', '--biorate', type=float, default=0,help='Biofouling Rate (mm/d): 0.0001, 0.0005 or 0.01')
    p.add_argument('-biodens', '--biodens', type=float, default=1.38,help='Biofouling Density (g/cm³): 1.05, 1.1, 1.35 or 1.38 (Fisher et al., 1983)')
    p.add_argument('-chunksize', '--chunksize', default='auto', help='chunksize for computation')
    p.add_argument('-track_kernels', '--track_kernels', type=bool, default=False, help='track_kernels')
    args = p.parse_args()
    print('These are the arguments: ' + str(args))
    p_advect(name=args.name, stype=args.stype, unbeach = args.unbeach, kukulka=args.kukulka, 
              kh=args.kh, wind=args.wind, stokes=args.stokes, dt=args.dt, biorate=args.biorate, 
              biodens=args.biodens, chunksize=args.chunksize, track_kernels=args.track_kernels, 
              arguments=args)
    
