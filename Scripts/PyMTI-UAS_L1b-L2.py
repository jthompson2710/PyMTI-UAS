##############################################################################
# Copyright 2024 University of Texas at Austin
# Author: Dr. James Thompson
# Created on Jun 6 2022
# This code calibrates data using calibration coeffiecnts, producing L1c and L2 data
# Updated on March 5 2024
##############################################################################
##############################################################################
# Functions
import numpy as np # version 1.19.2
import os # version 
import csv # version 
import shutil # version
import pandas # version 1.1.3 
import jdcal
import datetime
import matplotlib
import matplotlib.pyplot as plt
#from statsmodels import robust
from getListOfFiles import getListOfFiles
import natsort
import json
import base64
import array
from numpy import log, exp, zeros_like
from math import pi
from numpy import alltrue, any, arange, array_equal, asarray, diff, exp
from numpy import iscomplex, isnan, log, polyder, polyfit, polyval
from numpy import setdiff1d, sometrue, unique, var, where, zeros
from numpy import zeros_like, max, pi
import imreg_dft as ird
from osgeo import gdal # version 2.3.2
from TES_Drone import *
import time
#matplotlib.use('Agg') # use a non-interactive backend
#matplotlib.use('Qt5Agg') # use an 
##############################################################################
#camera variables
iTCam = ['TCam22','TCam23','TCam24','TCam25','TCam26']
iTCamrotate = [0, 0, 0, 1, 1]
FPAtemp0 = np.asarray([303.86,303.99,309.63,304.34,306.3], dtype=(np.float32)) #temperatures of FPA during FF calibration
ZeroRadiance = np.asarray([5.476,5.811,6.174,6.226,6.209,6.134], dtype=(np.float32)) #wavelength order 8.5,9.0,10.0,10.5,11.0,11.5 at 0 celsius
actual_rad_order = [1,2,0,4,3,5] 
wavelengths = [8.5,9.0,10.0,10.5,11.0,11.5]
allfolders = ['./Data/Iceland2020_HG_08242022_1/','./Data/Iceland2020_HG_08242022_2/','./Data/Iceland2020_HG_08242022_3/']
##############################################################################

for ff in range(len(allfolders)):
    
    folder = allfolders[ff]
    L1afolder = folder+'L1a/'
    L1cfolder = folder+'L1c/'
    l2_dir = folder+'L2/'
    
    #get list of all L1a data files in directories
    allfiles = getListOfFiles(L1afolder)
    #replace \\ with /
    allfiles = [sub.replace('\\', '/') for sub in allfiles] 
    #filter only txt files
    txtfiles = list(filter(lambda x: '.txt' in x, allfiles))
    
    TCam22_L1aRadianceAll=natsort.natsorted(list(filter(lambda x: 'TCam22' in x, txtfiles)))
    TCam23_L1aRadianceAll=natsort.natsorted(list(filter(lambda x: 'TCam23' in x, txtfiles)))
    TCam24_L1aRadianceAll=natsort.natsorted(list(filter(lambda x: 'TCam24' in x, txtfiles)))
    TCam25_L1aRadianceAll=natsort.natsorted(list(filter(lambda x: 'TCam25' in x, txtfiles)))
    TCam26_L1aRadianceAll=natsort.natsorted(list(filter(lambda x: 'TCam26' in x, txtfiles)))
    
    index22=[]
    index23=[]
    index24=[]
    index25=[]
    index26=[]
    
    for k in range(len(TCam22_L1aRadianceAll)):
        
        filename = TCam22_L1aRadianceAll[k][TCam22_L1aRadianceAll[k].rfind('/TCam22')+7:TCam22_L1aRadianceAll[k].find('.txt')]
        timestamp = filename[filename.rfind("_")+1:]
        timestamp_p500 = str(int(timestamp)+500)
        timestamp_m500 = str(int(timestamp)-500)
        filename_notime = filename[0:filename.rfind("_")+1]

        TCam22_L1aRadiance_filename = L1afolder+'Data/TCam22/TCam22'+filename_notime+timestamp+'.txt'
        TCam22_L1aRadiance_filename_p500 = L1afolder+'Data/TCam22/TCam22'+filename_notime+timestamp_p500+'.txt'
        TCam22_L1aRadiance_filename_m500 = L1afolder+'Data/TCam22/TCam22'+filename_notime+timestamp_m500+'.txt'
        if TCam22_L1aRadiance_filename in TCam22_L1aRadianceAll: index22.append(TCam22_L1aRadianceAll.index(TCam22_L1aRadiance_filename))
        elif TCam22_L1aRadiance_filename_p500 in TCam22_L1aRadianceAll: index22.append(TCam22_L1aRadianceAll.index(TCam22_L1aRadiance_filename_p500))
        elif TCam22_L1aRadiance_filename_m500 in TCam22_L1aRadianceAll: index22.append(TCam22_L1aRadianceAll.index(TCam22_L1aRadiance_filename_m500))
        else: index22.append(-1)

        TCam23_L1aRadiance_filename = L1afolder+'Data/TCam23/TCam23'+filename+'.txt'
        TCam23_L1aRadiance_filename_p500 = L1afolder+'Data/TCam23/TCam23'+filename_notime+timestamp_p500+'.txt'
        TCam23_L1aRadiance_filename_m500 = L1afolder+'Data/TCam23/TCam23'+filename_notime+timestamp_m500+'.txt'
        if TCam23_L1aRadiance_filename in TCam23_L1aRadianceAll: index23.append(TCam23_L1aRadianceAll.index(TCam23_L1aRadiance_filename))
        elif TCam23_L1aRadiance_filename_p500 in TCam23_L1aRadianceAll: index23.append(TCam23_L1aRadianceAll.index(TCam23_L1aRadiance_filename_p500))
        elif TCam23_L1aRadiance_filename_m500 in TCam23_L1aRadianceAll: index23.append(TCam23_L1aRadianceAll.index(TCam23_L1aRadiance_filename_m500))
        else: index23.append(-1)

        TCam24_L1aRadiance_filename = L1afolder+'Data/TCam24/TCam24'+filename+'.txt'
        TCam24_L1aRadiance_filename_p500 = L1afolder+'Data/TCam24/TCam24'+filename_notime+timestamp_p500+'.txt'
        TCam24_L1aRadiance_filename_m500 = L1afolder+'Data/TCam24/TCam24'+filename_notime+timestamp_m500+'.txt'
        if TCam24_L1aRadiance_filename in TCam24_L1aRadianceAll: index24.append(TCam24_L1aRadianceAll.index(TCam24_L1aRadiance_filename))
        elif TCam24_L1aRadiance_filename_p500 in TCam24_L1aRadianceAll: index24.append(TCam24_L1aRadianceAll.index(TCam24_L1aRadiance_filename_p500))
        elif TCam24_L1aRadiance_filename_m500 in TCam24_L1aRadianceAll: index24.append(TCam24_L1aRadianceAll.index(TCam24_L1aRadiance_filename_m500))
        else: index24.append(-1)

        TCam25_L1aRadiance_filename = L1afolder+'Data/TCam25/TCam25'+filename+'.txt'
        TCam25_L1aRadiance_filename_p500 = L1afolder+'Data/TCam25/TCam25'+filename_notime+timestamp_p500+'.txt'
        TCam25_L1aRadiance_filename_m500 = L1afolder+'Data/TCam25/TCam25'+filename_notime+timestamp_m500+'.txt'
        if TCam25_L1aRadiance_filename in TCam25_L1aRadianceAll: index25.append(TCam25_L1aRadianceAll.index(TCam25_L1aRadiance_filename))
        elif TCam25_L1aRadiance_filename_p500 in TCam25_L1aRadianceAll: index25.append(TCam25_L1aRadianceAll.index(TCam25_L1aRadiance_filename_p500))
        elif TCam25_L1aRadiance_filename_m500 in TCam25_L1aRadianceAll: index25.append(TCam25_L1aRadianceAll.index(TCam25_L1aRadiance_filename_m500))
        else: index25.append(-1)

        TCam26_L1aRadiance_filename = L1afolder+'Data/TCam26/TCam26'+filename+'.txt'
        TCam26_L1aRadiance_filename_p500 = L1afolder+'Data/TCam26/TCam26'+filename_notime+timestamp_p500+'.txt'
        TCam26_L1aRadiance_filename_m500 = L1afolder+'Data/TCam26/TCam26'+filename_notime+timestamp_m500+'.txt'
        if TCam26_L1aRadiance_filename in TCam26_L1aRadianceAll: index26.append(TCam26_L1aRadianceAll.index(TCam26_L1aRadiance_filename))
        elif TCam26_L1aRadiance_filename_p500 in TCam26_L1aRadianceAll: index26.append(TCam26_L1aRadianceAll.index(TCam26_L1aRadiance_filename_p500))
        elif TCam26_L1aRadiance_filename_m500 in TCam26_L1aRadianceAll: index26.append(TCam26_L1aRadianceAll.index(TCam26_L1aRadiance_filename_m500))
        else: index26.append(-1)

    for k in range(len(index22)):
        
        if ((index22[k]!=-1) and (index23[k]!=-1) and (index24[k]!=-1) and (index25[k]!=-1) and (index26[k]!=-1)):
    
            filename = TCam22_L1aRadianceAll[index22[k]][TCam22_L1aRadianceAll[index22[k]].rfind('/TCam22')+7:TCam22_L1aRadianceAll[index22[k]].find('.txt')]
            L1a24 = np.loadtxt(TCam24_L1aRadianceAll[index24[k]])
            L1a22 = np.loadtxt(TCam22_L1aRadianceAll[index22[k]])
            L1a23 = np.loadtxt(TCam23_L1aRadianceAll[index23[k]])
            L1a26 = np.loadtxt(TCam26_L1aRadianceAll[index26[k]])
            L1a25 = np.loadtxt(TCam25_L1aRadianceAll[index25[k]])
            
            L1c24 = L1a24
            ird23 = ird.similarity(L1c24, L1a23, numiter=3, constraints=dict(scale=[1.0,0]))
            L1c23 = ird23['timg']
            ird25 = ird.similarity(L1c24, L1a25, numiter=3, constraints=dict(scale=[1.0,0]))
            L1c25 = ird25['timg']
            ird26 = ird.similarity(L1c24, L1a26, numiter=3, constraints=dict(scale=[1.0,0]))
            L1c26 = ird26['timg']
            ird22 = ird.translation(L1c24, L1a22)
            L1c22 = ird.transform_img(L1a22, tvec=ird22['tvec'].round(4))
    
    #        iL1bRadPacket = zeros([1,5,120,160])
            iL1bRadPacket = np.stack([L1c24,L1c22,L1c23,L1c26,L1c25])
            if not os.path.exists(L1cfolder+'Rad/'):
                os.makedirs(L1cfolder+'Rad/')
            dst_filename = L1cfolder+'Rad/TIR_Rad_'+filename+'.tif'
            x_pixels = 160  # number of pixels in x
            y_pixels = 120  # number of pixels in y
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(dst_filename,x_pixels, y_pixels, 5,gdal.GDT_Float64)
            dataset.GetRasterBand(1).WriteArray(L1c24)
            dataset.GetRasterBand(2).WriteArray(L1c22)
            dataset.GetRasterBand(3).WriteArray(L1c23)
            dataset.GetRasterBand(4).WriteArray(L1c26)
            dataset.GetRasterBand(5).WriteArray(L1c25)
            dataset.FlushCache()
            dataset=None
            driver=None
    
#           TES Algorithm

            surfrad = iL1bRadPacket
            emisf_tes = zeros_like(iL1bRadPacket)
            
            ecw_tir = wavelengths
            tes_bands=[0,3,4]
            ecw_tes = [ecw_tir[i] for i in tes_bands]
            
            #set up variables for TES algorithm: skyr_tes,skyr_nontes, surfrad_tes, and
            #surfrad_nontes
            [surfrad_tes, surfrad_nontes] = init_TES(surfrad)
            [emisf, emisf_nontes, Ts, QAmap, wave_tes, wave_nontes]=TES_for_Drone(surfrad_tes, surfrad_nontes, ecw_tes, ecw_tir)
            #combine output data 
            emisf_tes0 = np.float32(emisf[0,:,:])
            emisf_tes[0,:,:] = emisf[0,:,:]
            emisf_tes1 = np.float32(emisf_nontes[0,:,:])
            emisf_tes[1,:,:] = emisf_nontes[0,:,:]
            emisf_tes2 = np.float32(emisf_nontes[1,:,:])
            emisf_tes[2,:,:] = emisf_nontes[1,:,:]
            emisf_tes3 = np.float32(emisf[1,:,:])
            emisf_tes[3,:,:] = emisf[1,:,:]
            emisf_tes4 = np.float32(emisf[2,:,:])
            emisf_tes[4,:,:] = emisf[2,:,:]
        
            Ts1 = np.float32(Ts)
            
            #make Emi output directory if not already exist
            if not os.path.exists(l2_dir+'TES/Emi/'):
                os.makedirs(l2_dir+'TES/Emi/')
            outputfilenameEmi = l2_dir+'TES/Emi/TES_Emi_'+filename
            EmiDriver = gdal.GetDriverByName('GTiff')
            EmiRaster = EmiDriver.Create(outputfilenameEmi+'.tif', xsize=x_pixels, ysize=y_pixels, bands=5, eType=gdal.GDT_Float32)
            EmiBand0 = EmiRaster.GetRasterBand(1).ReadAsArray()
            EmiBand0[:,:] = emisf_tes0
            EmiRaster.GetRasterBand(1).WriteArray(EmiBand0)
            EmiBand1 = EmiRaster.GetRasterBand(2).ReadAsArray()
            EmiBand1[:,:] = emisf_tes1
            EmiRaster.GetRasterBand(2).WriteArray(EmiBand1)
            EmiBand2 = EmiRaster.GetRasterBand(3).ReadAsArray()
            EmiBand2[:,:] = emisf_tes2
            EmiRaster.GetRasterBand(3).WriteArray(EmiBand2)
            EmiBand3 = EmiRaster.GetRasterBand(4).ReadAsArray()
            EmiBand3[:,:] = emisf_tes3
            EmiRaster.GetRasterBand(4).WriteArray(EmiBand3)
            EmiBand4 = EmiRaster.GetRasterBand(5).ReadAsArray()
            EmiBand4[:,:] = emisf_tes4
            EmiRaster.GetRasterBand(5).WriteArray(EmiBand4)                
            EmiRaster = None
            
            
            #make Emi output directory if not already exist
            if not os.path.exists(l2_dir+'TES/Temp/GTiff/'):
                os.makedirs(l2_dir+'TES/Temp/GTiff/')
            outputfilenameTemp = l2_dir+'TES/Temp/GTiff/TES_Temp_'+filename
            TempDriver = gdal.GetDriverByName('GTiff')
            TempRaster = TempDriver.Create(outputfilenameTemp+'.tif', xsize=x_pixels, ysize=y_pixels, bands=1, eType=gdal.GDT_Float32)
            TempBand = TempRaster.GetRasterBand(1).ReadAsArray()
            TempBand[:,:] = Ts1
            TempRaster.GetRasterBand(1).WriteArray(TempBand)
            TempRaster=None
    
            if not os.path.exists(l2_dir+'TES/Temp/PNG/'):
                os.makedirs(l2_dir+'TES/Temp/PNG/')
            outputfilenameTempimg = l2_dir+'TES/Temp/PNG/TES_Temp_'+filename
            # #show data
            plt.imshow(Ts1, cmap='plasma',vmin=280, vmax=550)
            plt.colorbar(label='Temperature (Kelvin)',shrink=0.8)
            plt.title('L2 Temperautre   '+  filename)
            # plt.savefig(outputfilenameTempimg+'.png', dpi=250, bbox_inches='tight', pad_inches=0)
            plt.close()
                    
            if not os.path.exists(l2_dir+'TES/Temp/TXT/'):
                os.makedirs(l2_dir+'TES/Temp/TXT/')
            outputfilenameTemptxt = l2_dir+'TES/Temp/TXT/TES_Temp_'+filename
            # np.savetxt(outputfilenameTemptxt+'.txt', Ts1, fmt='%.4f')

            if not os.path.exists(l2_dir+'TES/Temp/SfM/'):
                os.makedirs(l2_dir+'TES/Temp/SfM/')
            outputfilenameTempSfM = l2_dir+'TES/Temp/SfM/TES_Temp_'+filename
            # #show data
            plt.imshow(Ts1, cmap='plasma',vmin=280, vmax=550)
            plt.axis('off')
            # plt.savefig(outputfilenameTempSfM+'.png', dpi=250, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(filename+":  "+str(k)+" / "+str(len(index22)))

    # perge variables 
    folder = None
    L1afolder = None
    L1cfolder = None
    allfiles = None
    allfiles = None
    txtfiles = None
    TCam22_L1aRadianceAll=None
    TCam23_L1aRadianceAll=None
    TCam24_L1aRadianceAll=None
    TCam25_L1aRadianceAll=None
    TCam26_L1aRadianceAll=None
    index22=None
    index23=None
    index24=None
    index25=None
    index26=None
##############################################################################






