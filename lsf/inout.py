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


def write_lsf_to_fits(data, filepath, extname, version=None, clobber=False,
                      key_fields=None):
    """
    Parameters
    ----------
    key_fields : tuple of str, optional
        If given (e.g. ('order','segm')), performs a KEY-AWARE merge
        instead of the default blind positional overwrite: existing rows
        whose key matches a row in `data` are REPLACED; rows in `data`
        with no matching existing key are APPENDED (the table is resized
        to fit, via a plain fitsio .write() with no firstrow= -- fitsio
        correctly resizes/replaces the whole extension's content when
        called that way; this is not a per-row in-place patch, it's a
        full rewrite of the extension with the merged array).

        Without this (the default, None), the ORIGINAL behavior is
        unchanged: overwrite exactly the first len(data) rows in place,
        regardless of order/segment identity. That default is what
        callers without a natural row key (e.g. linelist, where
        overwrite-in-place on rerun is the desired behavior) still get.

        Per-order LSF construction (pixel_gp/velocity_gp/pixel_model/
        velocity_model, all of which have 'order'+'segm') should pass
        key_fields=('order','segm') -- otherwise, running for a NEW
        order just overwrites whatever's in rows [0, numseg) regardless
        of which order that data actually belonged to, silently
        destroying it. See conversation history for the concrete
        reproduction of this.
    """
    print(filepath)
    with FITS(filepath, mode='rw', clobber=clobber) as hdu:
        status = 'failed'
        try:
            existing = hdu[extname, version].read()
            exists = True
        except Exception:
            exists = False
            existing = None

        try:
            if not exists:
                hdu.write(data, extname=extname, extver=version)
                action = 'write'
            elif key_fields is None:
                existing_len = len(existing)
                if existing_len != len(data):
                    print(f"WARNING: '{extname}' version {version} already "
                         f"has {existing_len} rows, but the new data has "
                         f"{len(data)}. Overwriting the first {len(data)} "
                         f"rows in place — if {len(data)} < {existing_len}, "
                         f"{existing_len - len(data)} stale trailing row(s) "
                         f"from a previous run will remain. If that's not "
                         f"what you want, delete the file or pass "
                         f"clobber=True for a clean rewrite.")
                hdu[extname, version].write(data, firstrow=0)
                action = 'overwrite'
            else:
                combined, n_replaced, n_appended = _merge_rows_by_key(
                    existing, data, key_fields
                )
                print(f"'{extname}' version {version}: merging by "
                     f"{key_fields} -- {n_replaced} existing row(s) "
                     f"replaced, {n_appended} new row(s) appended "
                     f"({len(existing)} -> {len(combined)} total rows).")
                hdu[extname, version].write(combined)   # no firstrow= -> resizes
                action = 'merge'
            status = 'done'
        except Exception:
            hdu.write(data, extname=extname, extver=version)
            action = 'write (fallback)'
            status = 'done'
        finally:
            hdu.close()
            print(f"Data {action} to {filepath} {status}.")
    return None


def _merge_rows_by_key(existing, new, key_fields):
    """
    existing, new : structured numpy arrays with the same dtype
    key_fields    : tuple of field names identifying a row uniquely
                    (e.g. ('order','segm'))

    Returns (combined, n_replaced, n_appended):
      combined   : existing rows NOT matched by any key in `new`,
                   followed by every row of `new` (so a row being
                   updated moves to wherever it appears in `new`,
                   not staying at its old position -- this is a
                   feature, not just a side effect: it keeps rows for
                   one order contiguous instead of scattering an
                   updated order's rows back into wherever they were
                   first written).
      n_replaced : rows in `new` whose key already existed in `existing`
      n_appended : rows in `new` whose key is new
    """
    def keys_of(arr):
        return set(zip(*[arr[f].tolist() for f in key_fields]))

    new_keys = keys_of(new)
    if len(existing) == 0:
        keep_mask = np.zeros(0, dtype=bool)
    else:
        existing_keys = list(zip(*[existing[f].tolist() for f in key_fields]))
        keep_mask = np.array([k not in new_keys for k in existing_keys])

    existing_keys_set = keys_of(existing) if len(existing) else set()
    new_row_keys = list(zip(*[new[f].tolist() for f in key_fields]))
    n_replaced = sum(1 for k in new_row_keys if k in existing_keys_set)
    n_appended = len(new) - n_replaced

    kept_existing = existing[keep_mask] if len(existing) else existing
    combined = np.concatenate([kept_existing, new])
    return combined, n_replaced, n_appended


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


