#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:51:13 2023

@author: dmilakov
"""

import os
from fitsio import FITS
import harps.settings as hs
import numpy as np

def read_lsf_from_fits(filepath,extname,version):
    # checks whether the LSF fits file exists and reads it
    # exists = hio.fits_exists('lsf',spec.filepath)
    exists = os.path.exists(filepath)
    # lsf_filepath = hio.get_fits_path('lsf',spec.filepath)
    
    if exists:
        with FITS(filepath) as hdul:
            try:
                hasdata = hdul[extname,version].has_data()
            except:
                hasdata = False
            if hasdata:
                lsf2d = hdul[extname,version].read()
            else:
                print(f'Did not find the extension {extname}, version {version}. The file contains:')
                print(hdul)
                raise Exception
    else:
        raise Exception
    return lsf2d



def write_lsf_to_fits(data,filepath,extname,version=None,clobber=False):
    print(filepath)
    # dirpath=dirpath if dirpath is not None else hs.get_dirname('lsf')
    # filepath = os.path.join(dirpath,filename)
    with FITS(filepath,mode='rw',clobber=clobber) as hdu:
        status = 'failed'
        # extver_hasdata = False
        # try:
        #     hdu[extname,version].has_data()
        #     extvar_hasdata = True
        # except:
        #     pass
        
        # if extvar_hasdata & overwrite:
        #     expression = helper_get_expression(data)
        #     rows2del = hdu[extname,version].where()
        
        
        try:
            hdu[extname,version].append(data)
            action = 'append'
            status = 'done'
        except:
            hdu.write(data,extname=extname,extver=version)
            action = 'write'
            status = 'done'
        finally:
            hdu.close()
            print(f"Data {action} to {filepath} {status}.")
    return None


def convert_version(iteration,interpolate,model_scatter):
    assert iteration>0 and iteration<10
    int_iter = int(iteration)
    int_intr = int(interpolate)
    int_mosc = int(model_scatter)
    version  = int(f"{int_iter:1d}{int_intr:1d}{int_mosc:1d}")
    return version

def copy_linelist_inplace(filepath,new_ver):
    return copy_extension_inplace(filepath, extname='linelist',new_ver=new_ver,
                                  action='ignore')
def make_extension(filepath,extname,new_ver,shape,dtype=None):
    with FITS(filepath,mode='rw',clobber=False) as hdu:
        # break if exists
        extver_exists = False
        success = False
        try:
            extver_exists = hdu[extname,new_ver].has_data()
        except:
            pass
        # print(f"exists = {extver_exists}")
        if extver_exists:
            status = "NOT DONE (already exists)"
        else:
            dtype = dtype if dtype is not None else 'float32'
            data = np.zeros(shape=shape,dtype=dtype)
            hdu.write(data=data,extname=extname,extver=new_ver)
            
            # hdu['linelist',newver].write_comment(f'Copied from {oldver}')
            status = "DONE"
            success = True
    message = f"Copying {extname} in {filepath}"
    print(f"{message} {status}")
    return success
def copy_extension_inplace(filepath,extname,new_ver,action='ignore'):
    
    with FITS(filepath,mode='rw',clobber=False) as hdu:
        # break if exists
        extver_exists = False
        success = False
        try:
            extver_exists = hdu[extname,new_ver].has_data()
        except:
            pass
        # print(f"exists = {extver_exists}")
        if extver_exists:
            if action=='ignore':
                status = "NOT DONE (already exists)"
            elif action=='make':
                status = "CREATED EMPTY"
        else:
            print(extname,new_ver)
            # print(newitem, newver)
            llist_hdu = hdu[extname]
            # print(llist_hdu)
            data      = llist_hdu.read()
            header    = llist_hdu.read_header()
            header['EXTVER']=new_ver
            hdu.write(data=data,header=header,
                      extname=extname,extver=new_ver)
            
            # hdu['linelist',newver].write_comment(f'Copied from {oldver}')
            status = "DONE"
            success = True
            
            
    message = f"Copying {extname} {new_ver} in {filepath}"
    print(f"{message} {status}")
    return success



def copy_extension_to_new(infile,outfile,extname,extver,new_extver='same',
                          clobber=False):
    with FITS(infile,mode='rw',clobber=False) as hdu_in:
        # break if exists
        item = (extname,extver) if extver is not None else extname
        print(item)
        try:
            extver_exists = hdu_in[item].has_data()
        except:
            pass
        # print(f"exists = {extver_exists}")
        # print(newitem, newver)
        llist_hdu = hdu_in[item]
        # print(llist_hdu)
        data      = llist_hdu.read()
        header    = llist_hdu.read_header()
    with FITS(outfile,'rw',clobber=clobber) as hdu_out:
        new_extver = new_extver if new_extver != 'same' else extver
        header['EXTVER']=new_extver
        hdu_out.write(data=data,header=header,
                  extname=extname,extver=1)
        
        # hdu['linelist',newver].write_comment(f'Copied from {oldver}')
        status = "DONE"
        success = True
            
    message = f"Copying {extname} {extver} from {infile} to {outfile}"
    print(f"{message} {status}")
    return success