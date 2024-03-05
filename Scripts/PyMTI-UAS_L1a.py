##############################################################################
# Copyright 2024 University of Texas at Austin
# Author: Dr. James Thompson
# Created on Oct 29 2021
# This code calibrates data using calibration coeffiecnts, producing L1a data
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
matplotlib.use('Agg') # use a non-interactive backend
#matplotlib.use('Qt5Agg') # use an 
##############################################################################

#camera variables
iTCam = ['TCam22','TCam23','TCam24','TCam25','TCam26']
iTCamrotate = [0, 0, 0, 1, 1]
FPAtemp0 = np.asarray([303.86,303.99,309.63,304.34,306.3], dtype=(np.float32)) #temperatures of FPA during FF calibration
ZeroRadiance = np.asarray([5.476,5.811,6.174,6.226,6.209,6.134], dtype=(np.float32)) #wavelength order 8.5,9.0,10.0,10.5,11.0,11.5 at 0 celsius
actual_rad_order = [1,2,0,4,3,5] 
wavelengths = [8.5,9.0,10.0,10.5,11.0,11.5]
califolder = './Calibration/Constants/Tlin/HighGain/'
temporalfolder = '/Calibration/Constants/Temporal/Tlin/HighGain/'
allfolders = ['./Data/Iceland2020_HG_08242022_1/','./Data/Iceland2020_HG_08242022_2/','./Data/Iceland2020_HG_08242022_3/']
##############################################################################

for ff in range(len(allfolders)):
    folder = allfolders[ff]
    outfolder = folder+'L1a/'

    for it in range(len(iTCam)):
        all_c = np.load(califolder+iTCam[it]+'all_temporal_cali_c.npy')
        vars()[iTCam[it]+'temporaldata_quadfit'] = np.load(temporalfolder+iTCam[it]+'temporaldata_quadfit.npy')
        vars()[iTCam[it]+'temporalarray0'] = np.zeros((120, 160), np.float32)
        for r in range(0, 120):
            for c in range(0, 160):
                vars()[iTCam[it]+'temporalarray0'][r, c] = np.polyval(vars()[iTCam[it]+'temporaldata_quadfit'][:, r, c], FPAtemp0[it])
    
        
        #get list of all files in directories
        allfiles = getListOfFiles(folder+iTCam[it]+'/')
        #replace \\ with /
        allfiles = [sub.replace('\\', '/') for sub in allfiles] 
        #filter only txt files
        txtfiles = list(filter(lambda x: '.json' in x, allfiles))
        #filter based on camera and temperautre setting
        vars()[iTCam[it]+'files'] = natsort.natsorted(list(filter(lambda x: iTCam[it] in x, txtfiles)))
        
        for k in range(len(vars()[iTCam[it]+'files'])):
            file1  = vars()[iTCam[it]+'files'][k]
            
            with open(file1) as json_file:
                data0 = json.load(json_file)
            
            dec_tel1 = base64.b64decode(data0["telemetry"])
            tel1 = array.array('H', dec_tel1)    
            FPAtemp1 = np.asarray((tel1[24] / 100))
            temporalarray1 = np.zeros((120, 160), np.float32)
            for r in range(0, 120):
                for c in range(0, 160):
                    temporalarray1[r, c] = np.polyval(vars()[iTCam[it]+'temporaldata_quadfit'][:, r, c], FPAtemp1)
            temporalarraydifference = vars()[iTCam[it]+'temporalarray0'] - temporalarray1
                
            
            dec_rad0 = base64.b64decode(data0["radiometric"])
            ra0 = array.array('H', dec_rad0)
            dataarray0 = np.zeros((120, 160), np.float32)
            for r in range(0, 120):
                for c in range(0, 160):
                    val0 = (ra0[(r * 160) + c])
                    dataarray0[r, c] = val0
            
            #rotates images which were incorrectly orientated - 180 degrees
            if ('TCam25' in file1) or ('TCam26' in file1):
                a2 = np.zeros((120, 160), np.float32)
                for r in range(0, 120):
                    for c in range(0, 160):
                        a2[r,160-1-c] = dataarray0[120-1-r,c]
                dataarray0 = a2
            
            dataarray0 = dataarray0 - temporalarray1
            
            datashape = dataarray0.shape
            L1Radiance = np.empty([datashape[0],datashape[1]],dtype=np.float32)
            for i in range(datashape[0]):
                for j in range(datashape[1]):     
                    cali_c = all_c[i,j,:]
                    radiance1 = np.polyval(cali_c, dataarray0[i,j])
                    L1Radiance[i,j] = radiance1
                    
            L1Radiancemin = np.min(L1Radiance)
            L1Radiance = (L1Radiance - L1Radiancemin) + ZeroRadiance[actual_rad_order[it]]
            
            if iTCamrotate[it] == 1:
                L1Radiance180 = np.empty([datashape[0],datashape[1]],dtype=np.float32)
                for r in range(0, datashape[0]):
                    for c in range(0, datashape[1]):
                        L1Radiance180[r,datashape[1]-1-c] = L1Radiance[datashape[0]-1-r,c]
                L1Radiance = L1Radiance180
            
            filename = file1[file1.rfind('/')+1:file1.find('.json')]
            timestamp = filename[filename.rfind('_')+1:]
            # #show data
            plt.imshow(L1Radiance, cmap='coolwarm')
            plt.colorbar(label='Radiance (Wm$^{-2}$sr$^{-1}$$\mu$m$^{-1}$)')
            plt.title('L1 Radiance ('+ str(wavelengths[actual_rad_order[2]]) + 'um)  '+  timestamp)
            #build directory
            if not os.path.exists(outfolder+'/Images/'+iTCam[it]+'/'):
                os.makedirs(outfolder+'/Images/'+iTCam[it]+'/')
            plt.savefig(outfolder+'/Images/'+iTCam[it]+'/'+filename+'.png', dpi=250, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            if not os.path.exists(outfolder+'/Data/'+iTCam[it]+'/'):
                os.makedirs(outfolder+'/Data/'+iTCam[it]+'/')
            np.savetxt(outfolder+'/Data/'+iTCam[it]+'/'+filename+'.txt', L1Radiance, fmt='%.4f')

##############################################################################

            
