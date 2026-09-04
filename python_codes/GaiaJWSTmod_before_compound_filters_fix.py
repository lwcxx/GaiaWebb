#!/usr/bin/env python

import sys
import os
import subprocess
import warnings
import re
import shutil

import itertools
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy import stats
from math import log10, floor
import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.visualization import (ImageNormalize, ManualInterval)
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astroquery.mast import Observations

import reproject

def round_significant(x, ex, sig=1):
   """
   This routine returns a quantity rounded to its error significan figures.
   """

   significant = sig-int(floor(log10(abs(ex))))-1

   return round(x, significant), round(ex, significant)


def manual_select_from_cmd(color, mag):
   """
   Select stars based on their membership probabilities and cmd position
   """
   from matplotlib.path import Path
   from matplotlib.widgets import LassoSelector

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.15):
         self.canvas = ax.figure.canvas
         self.collection = collection
         self.alpha_other = alpha_other

         self.xys = collection.get_offsets()
         self.Npts = len(self.xys)

         # Ensure that we have separate colors for each object
         self.fc = collection.get_facecolors()
         if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
         elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

         lineprops = {'color': 'r', 'linewidth': 1, 'alpha': 0.8}
         self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
         self.ind = []

      def onselect(self, verts):
         path = Path(verts)
         self.ind = np.nonzero(path.contains_points(self.xys))[0]
         self.selection = path.contains_points(self.xys)
         self.fc[:, -1] = self.alpha_other
         self.fc[self.ind, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

      def disconnect(self):
         self.lasso.disconnect_events()
         self.fc[:, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

   help =  '----------------------------------------------------------------------------\n'\
           'Please, select likely member stars in the color-magnitude diagram (CMD).\n'\
           '----------------------------------------------------------------------------\n'\
           '- Look in the CMD for any sequence formed by possible member stars.\n'\
           '- Click and drag your cursor to draw a region around these stars.\n'\
           '- On release, the stars contained within the drawn region will be selected.\n'\
           '- Repeat if necessary until you are satisfied with the selection.\n'\
           '- Press enter once finished and follow the instructions in the terminal.\n'\
           '----------------------------------------------------------------------------'

   print('\n'+help+'\n')

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(color, mag, c = '0.2', s=1)

   try:
      ax.set_xlim(np.nanmin(color)-0.1, np.nanmax(color)+0.1)
      ax.set_ylim(np.nanmax(mag)+0.05, np.nanmin(mag)-0.05)
   except:
      pass

   ax.grid()
   ax.set_xlabel(color.name.replace("_wmean","").replace("_mean",""))
   ax.set_ylabel(mag.name.replace("_wmean","").replace("_mean",""))

   selector = SelectFromCollection(ax, pts)

   def accept(event):
      if event.key == "enter":
         selector.disconnect()
         plt.close('all')

   fig.canvas.mpl_connect("key_press_event", accept)
   fig.suptitle("Click and move your cursor to select member stars. Press enter to accept.", fontsize=11)
   
   plt.tight_layout()
   plt.show()

   input("Please, press enter to continue.")

   try:
      return selector.selection
   except:
      return [True]*len(mag)



def manual_select_from_pm(pmra, pmdec):
   """
   Select stars based on their membership probabilities and VPD position
   """
   from matplotlib.path import Path
   from matplotlib.widgets import LassoSelector

   class SelectFromCollection(object):
      """
      Select indices from a matplotlib collection using `LassoSelector`.
      """
      def __init__(self, ax, collection, alpha_other=0.15):
         self.canvas = ax.figure.canvas
         self.collection = collection
         self.alpha_other = alpha_other

         self.xys = collection.get_offsets()
         self.Npts = len(self.xys)

         # Ensure that we have separate colors for each object
         self.fc = collection.get_facecolors()
         if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
         elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

         lineprops = {'color': 'r', 'linewidth': 1, 'alpha': 0.8}
         self.lasso = LassoSelector(ax, onselect=self.onselect, lineprops=lineprops)
         self.ind = []

      def onselect(self, verts):
         path = Path(verts)
         self.ind = np.nonzero(path.contains_points(self.xys))[0]
         self.selection = path.contains_points(self.xys)
         self.fc[:, -1] = self.alpha_other
         self.fc[self.ind, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

      def disconnect(self):
         self.lasso.disconnect_events()
         self.fc[:, -1] = 1
         self.collection.set_facecolors(self.fc)
         self.canvas.draw_idle()

   help =  '----------------------------------------------------------------------------\n'\
           'Please, select likely member stars in the vector-point diagram (VPD).\n'\
           '----------------------------------------------------------------------------\n'\
           '- Look in the VPD for any clump formed by possible member stars.\n'\
           '- Click and drag your cursor to draw a region around these stars.\n'\
           '- On release, the stars contained within the drawn region will be selected.\n'\
           '- Repeat if necessary until you are satisfied with the selection.\n'\
           '- Press enter once finished and follow the instructions in the terminal.\n'\
           '----------------------------------------------------------------------------'

   print('\n'+help+'\n')

   subplot_kw = dict(autoscale_on = False)
   fig, ax = plt.subplots(subplot_kw = subplot_kw)

   pts = ax.scatter(pmra, pmdec, c = '0.2', s=1)

   try:
      margin = 2*(np.nanstd(pmra)+np.nanstd(pmdec))/2
      ax.set_xlim(np.nanmedian(pmra)-margin, np.nanmedian(pmra)+margin)
      ax.set_ylim(np.nanmedian(pmdec)-margin, np.nanmedian(pmdec)+margin)
   except:
      pass

   ax.grid()
   ax.set_xlabel(r'$\mu_{\alpha\star}$')
   ax.set_ylabel(r'$\mu_{\delta}$')

   selector = SelectFromCollection(ax, pts)

   def accept(event):
      if event.key == "enter":
         selector.disconnect()
         plt.close('all')

   fig.canvas.mpl_connect("key_press_event", accept)
   fig.suptitle("Click and move your cursor to select member stars. Press enter to accept.", fontsize=11)
   
   plt.tight_layout()
   plt.show()

   input("Please, press enter to continue.")

   try:
      return selector.selection
   except:
      return [True]*len(mag)


def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
   """
   Calculate the corrected flux excess factor for the input Gaia EDR3 data.
   """

   if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
       bp_rp = np.float64(bp_rp)
       phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
   
   if bp_rp.shape != phot_bp_rp_excess_factor.shape:
       raise ValueError('Function parameters must be of the same shape!')
   
   do_not_correct = np.isnan(bp_rp)
   bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
   greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
   redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)
   
   correction = np.zeros_like(bp_rp)
   correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange],2)
   correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange],2) - 0.005879*np.power(bp_rp[greenrange],3)
   correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
   
   return phot_bp_rp_excess_factor - correction


def clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = False):
   """
   Select stars with good astrometry in Gaia.
   """

   labels_ruwe = ruwe <= 1.4
   labels_harmonic_amplitude = ipd_gof_harmonic_amplitude <= 0.2                                    # Reject blended transits Fabricius et al. (2020)
   labels_visibility = visibility_periods_used >= 9                                                 # Lindengren et al. (2020)
   labels_excess_noise = astrometric_excess_noise_sig <= 2.0                                        # Lindengren et al. (2020)

   labels_astrometric = labels_ruwe & labels_harmonic_amplitude & labels_visibility & labels_excess_noise

   if use_5p:
      labels_params_solved = astrometric_params_solved == 31                                        # 5p parameters solved Brown et al. (2020)
      labels_astrometric = labels_astrometric & labels_params_solved

   return labels_astrometric


def clean_photometry(gmag, corrected_flux_excess_factor, sigma_flux_excess_factor = 3):
   """
   This routine select stars based on their flux_excess_factor. Riello et al.2020
   """
   from matplotlib.path import Path

   def sigma_corrected_C(gmag, sigma_flux_excess_factor):
      return sigma_flux_excess_factor*(0.0059898 + 8.817481e-12 * gmag ** 7.618399)

   mag_nodes = np.linspace(np.min(gmag)-0.1, np.max(gmag)+0.1, 100)
   
   up = [(gmag, sigma) for gmag, sigma in zip(mag_nodes, sigma_corrected_C(mag_nodes, sigma_flux_excess_factor))]
   down = [(gmag, sigma) for gmag, sigma in zip(mag_nodes[::-1], sigma_corrected_C(mag_nodes, -sigma_flux_excess_factor)[::-1])]

   path_C = Path(up+down, closed=True)
   
   labels_photometric = path_C.contains_points(np.array([gmag, corrected_flux_excess_factor]).T)

   return labels_photometric


def pre_clean_data(phot_g_mean_mag, corrected_flux_excess_factor, ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, sigma_flux_excess_factor = 3, use_5p = False):
   """
   This routine cleans the Gaia data from astrometrically and photometric bad measured stars.
   """
   
   labels_photometric = clean_photometry(phot_g_mean_mag, corrected_flux_excess_factor, sigma_flux_excess_factor = sigma_flux_excess_factor)
   
   labels_astrometric = clean_astrometry(ruwe, ipd_gof_harmonic_amplitude, visibility_periods_used, astrometric_excess_noise_sig, astrometric_params_solved, use_5p = use_5p)
   
   return labels_astrometric & labels_photometric


def remove_jobs():
   """
   This routine removes jobs from the Gaia archive server.
   """

   list_jobs = []
   for job in Gaia.list_async_jobs():
      list_jobs.append(job.get_jobid())
   
   Gaia.remove_jobs(list_jobs)


def gaia_query(Gaia, query, min_gmag, max_gmag, save_individual_queries, load_existing, name, n, n_total):
   """
   This routine launch the query to the Gaia archive.
   """

   query = query + " AND (phot_g_mean_mag > %.4f) AND (phot_g_mean_mag <= %.4f)"%(min_gmag, max_gmag)

   individual_query_filename = './%s/Gaia/individual_queries/%s_G_%.4f_%.4f.csv'%(name, name, min_gmag, max_gmag)

   if os.path.isfile(individual_query_filename) and load_existing:
      result = pd.read_csv(individual_query_filename)

   else:
      import time as _time

      mirrors = [
         ("https://gea.esac.esa.int/tap-server/tap", "async"),
         ("https://gaia.ari.uni-heidelberg.de/tap",  "async"),
      ]
      # If the caller already pointed at Heidelberg, try it first.
      tap_url = getattr(Gaia, 'TAP_URL', '')
      if "heidelberg" in tap_url.lower():
         mirrors = mirrors[::-1]

      last_exc = None
      result = None
      for mirror_url, mode in mirrors:
         for attempt in range(3):
            try:
               Gaia.TAP_URL = mirror_url
               if mode == "async":
                  job = Gaia.launch_job_async(query)
               else:
                  job = Gaia.launch_job(query)
               result = job.get_results()
               try:
                  Gaia.remove_jobs([job.jobid])
               except Exception:
                  pass
               result = result.to_pandas()
               last_exc = None
               break
            except Exception as exc:
               last_exc = exc
               wait = 10 * (attempt + 1)
               print('\nWARNING: Gaia query attempt %i failed (%s). Retrying in %is...' % (attempt + 1, exc, wait))
               _time.sleep(wait)
         if result is not None:
            break

      if result is None:
         raise RuntimeError(
            'All Gaia query attempts failed. Last error: %s' % last_exc
         )

      if save_individual_queries:
         result.to_csv(individual_query_filename, index = False)

   print('\n')
   print('----------------------------')
   print('Table %i of %i: %i stars'%(n, n_total, len(result)))
   print('----------------------------')

   return result, query


def get_mag_bins(min_mag, max_mag, area, mag = None):
   """
   This routine generates logarithmic spaced bins for G magnitude.
   """

   num_nodes = np.max((1, np.round( ( (max_mag - min_mag) * max_mag ** 2 * area)*5e-5)))

   bins_mag = (1.0 + max_mag - np.logspace(np.log10(1.), np.log10(1. + max_mag - min_mag), num = int(num_nodes), endpoint = True))

   return bins_mag


def gaia_multi_query_run(args):
   """
   This routine pipes gaia_query into multiple threads.
   """

   return gaia_query(*args)


def columns_n_conditions(source_table, astrometric_cols, photometric_cols, quality_cols, ra, dec, width = 1.0, height = 1.0):

   """
   This routine generates the columns and conditions for the query.
   """

   if 'dr3' in source_table:
      if 'ruwe' not in quality_cols:
         quality_cols = 'ruwe' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
   elif 'dr2' in source_table:
      if 'astrometric_n_good_obs_al' not in quality_cols:
         quality_cols = 'astrometric_n_good_obs_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'astrometric_chi2_al' not in quality_cols:
         quality_cols = 'astrometric_chi2_al' +  (', ' + quality_cols if len(quality_cols) > 1 else '')
      if 'phot_bp_rp_excess_factor' not in quality_cols:
         quality_cols = 'phot_bp_rp_excess_factor' +  (', ' + quality_cols if len(quality_cols) > 1 else '')

   conditions = "CONTAINS(POINT('ICRS',"+source_table+".ra,"+source_table+".dec),BOX('ICRS',%.8f,%.8f,%.8f,%.8f))=1"%(ra, dec, width, height)

   columns = (", " + astrometric_cols if len(astrometric_cols) > 1 else '') + (", " + photometric_cols if len(photometric_cols) > 1 else '') +  (", " + quality_cols if len(quality_cols) > 1 else '')

   query = "SELECT source_id " + columns + " FROM " + source_table + " WHERE " + conditions

   return query, quality_cols
   

def use_processors(n_processes):

   """
   This routine finds the number of available processors in your machine
   """

   from multiprocessing import cpu_count

   available_processors = cpu_count()

   n_processes = n_processes % (available_processors+1)

   if n_processes == 0:
      n_processes = 1
      print('WARNING: Found n_processes = 0. Falling back to default single-threaded execution (n_processes = 1).')

   return n_processes


def incremental_query(query, area, min_gmag = 10.0, max_gmag = 19.5, n_processes = 1, save_individual_queries = False, load_existing = False, name = 'output'):

   """
   This routine search the Gaia archive and downloads the stars using parallel workers.
   """

   print("\n---------------------")
   print("Downloading Gaia data")
   print("---------------------")

   from astroquery.gaia import Gaia
   from multiprocessing import Pool, cpu_count

   # Mirror fallback logic
   Gaia.TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
   try:
      # Simple query to check connectivity
      Gaia.launch_job("SELECT TOP 1 source_id FROM gaiadr3.gaia_source")
   except:
      print("\nWARNING: ESA Gaia Archive unavailable (HTTP 503/504). Falling back to Heidelberg mirror.")
      Gaia.TAP_URL = "https://gaia.ari.uni-heidelberg.de/tap"
      Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # Ensure table name matches mirror

   mag_nodes = get_mag_bins(min_gmag, max_gmag, area)
   n_total = len(mag_nodes)
   
   if (n_total > 1) and (n_processes != 1):

      print("Executing %s jobs."%(n_total-1))

      pool = Pool(int(np.min((n_total, 20, n_processes*2))))

      args = []
      for n, node in enumerate(range(n_total-1)):
         args.append((Gaia, query, mag_nodes[n+1], mag_nodes[n], save_individual_queries, load_existing, name, n, n_total))

      tables_gaia_queries = pool.map(gaia_multi_query_run, args)

      tables_gaia = [results[0] for results in tables_gaia_queries]
      queries = [results[1] for results in tables_gaia_queries]

      result_gaia = pd.concat(tables_gaia)

      pool.close()

   else:
      result_gaia, queries = gaia_query(Gaia, query, min_gmag, max_gmag, save_individual_queries, load_existing, name, 1, 1)

   return result_gaia, queries


def plot_fields(Gaia_table, obs_table, JWST_path, use_only_good_gaia = False, min_stars_alignment = 5, no_plots = False, name = 'test.png'):
   """
   This routine plots the fields and select Gaia stars within them.
   """

   from matplotlib.patheffects import withStroke
   from matplotlib.patches import (Polygon, Patch)
   from matplotlib.collections import PatchCollection
   from matplotlib.lines import Line2D
   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point

   def deg_to_hms(lat, even = True):
      from astropy.coordinates import Angle
      angle = Angle(lat, unit = u.deg)
      if even:
         if lat%30:
            string =''
         else:
            string = angle.to_string(unit=u.hour)
      else:
         string = angle.to_string(unit=u.hour)
      return string

   def deg_to_dms(lon):
      from astropy.coordinates import Angle
      angle = Angle(lat, unit = u.deg)
      string = angle.to_string(unit=u.degree)
      return string

   def coolwarm(filter, alpha = 1):
      color = [rgb for rgb in plt.cm.coolwarm(int(255 * (float(filter) - 555) / (850-555)))]
      color[-1] = alpha
      return color

   # If no_plots == False, then python opens a plotting device
   if no_plots == False:
      fig, ax = plt.subplots(1,1, figsize = (5., 4.75))

   if use_only_good_gaia:
      Gaia_table_count = Gaia_table[Gaia_table.clean_label == True]
   else:
      Gaia_table_count = Gaia_table

   patches = []
   fields_data = []
   bad_patches = []
   bad_fields_data = []
   
   # We rearrange obs_table for legibility
   obs_table = obs_table.sort_values(by =['s_ra', 's_dec', 'proposal_id', 'obsid'], ascending = False).reset_index(drop=True)
   #print("obs_table (not empty)", obs_table)
   print(obs_table.keys()) 
   #print(obs_table['s_ra'])
   #print(obs_table['s_dec'])
   for index_obs, (s_ra, s_dec, footprint_str, obsid, filter, t_bl, obs_id) in obs_table.loc[:, ['s_ra', 's_dec', 's_region', 'obsid', 'filters', 't_baseline', 'obs_id']].iterrows():
      cli_progress_test(index_obs+1, len(obs_table))
    
      idx_Gaia_in_field = []
      gaia_stars_per_poly = [] 
      list_coo = footprint_str.split('POLYGON')[1::]

      for poly in list_coo:
         try:
            poly = list(map(float, poly.split()))
         except:
            poly = list(map(float, poly.split('J2000')[1::][0].split()))

         tuples_list = [(ra % 360, dec) for ra, dec in zip(poly[0::2], poly[1::2])]

         # Make sure the field is complete. With at least 4 vertices.

         if len(tuples_list) >= 4:
            polygon = Polygon(tuples_list, closed=True)
            
            # Check if the set seems downloaded
            if os.path.isfile(JWST_path+'mastDownload/JWST/'+obs_id+'/'+obs_id+'_cal.fits'):
               fc = [0,1,0,0.2]
            else:
               fc = [1,1,1,0.2]

            footprint =  shap_polygon(tuples_list)

            star_counts = 0
            for idx, ra, dec in zip(Gaia_table_count.index, Gaia_table_count.ra, Gaia_table_count.dec):
               if Point(ra, dec).within(footprint):
                  idx_Gaia_in_field.append(idx)
                  star_counts += 1

            gaia_stars_per_poly.append(star_counts)
            
            
            if star_counts >= min_stars_alignment: 
               patches.append(polygon)
               fields_data.append([index_obs, round(s_ra, 2), round(s_dec, 2), filter, coolwarm(float(filter.replace(r'F', '').replace(r'W', '').replace(r'N', '').replace(r'M', '').replace('LP', '').replace('P', ''))), t_bl, fc, sum(gaia_stars_per_poly)])
            else:
               bad_patches.append(polygon)
               bad_fields_data.append([round(s_ra, 2), round(s_dec, 2), filter, coolwarm(float(filter.replace(r'F', '').replace(r'W', '').replace(r'N', '').replace(r'M', '').replace('LP', '').replace('P', ''))), fc, sum(gaia_stars_per_poly)])
   print('\n')

   fields_data = pd.DataFrame(data = fields_data, columns=['Index_obs', 'ra', 'dec', 'filter', 'filter_color', 't_baseline', 'download_color', 'gaia_stars_per_obs'])
   bad_fields_data = pd.DataFrame(data = bad_fields_data, columns=['ra', 'dec', 'filter', 'filter_color', 'download_color', 'gaia_stars_per_obs'])

   # Select only the observations with enough Gaia stars
   if (not fields_data.empty) and (not obs_table.empty):
      obs_table = obs_table.iloc[fields_data.Index_obs, :].reset_index(drop=True)
      obs_table['gaia_stars_per_obs'] = fields_data.gaia_stars_per_obs
   elif (not obs_table.empty):
      obs_table['gaia_stars_per_obs'] = bad_fields_data.gaia_stars_per_obs

   if no_plots == False:

      try:
         ra_lims = [min(Gaia_table.ra.max(), fields_data.ra.max()+0.2/np.cos(np.deg2rad(fields_data.dec.mean()))), max(Gaia_table.ra.min(), fields_data.ra.min()-0.2/np.cos(np.deg2rad(fields_data.dec.mean())))]
         dec_lims = [max(Gaia_table.dec.min(), fields_data.dec.min()-0.2), min(Gaia_table.dec.max(), fields_data.dec.max()+0.2)]
      except:
         ra_lims = [Gaia_table.ra.max(), Gaia_table.ra.min()]
         dec_lims = [Gaia_table.dec.min(), Gaia_table.dec.max()]

      bpe = PatchCollection(bad_patches, alpha = 0.1, ec = 'None', fc = bad_fields_data.download_color, antialiased = True, lw = 1, zorder = 2)
      bpf = PatchCollection(bad_patches, alpha = 1, ec = bad_fields_data.filter_color, fc = 'None', antialiased = True, lw = 1, zorder = 3, hatch='/////')

      ax.add_collection(bpe)
      ax.add_collection(bpf)

      pe = PatchCollection(patches, alpha = 0.1, ec = 'None', fc = fields_data.download_color, antialiased = True, lw = 1, zorder = 2)
      pf = PatchCollection(patches, alpha = 1, ec = fields_data.filter_color, fc = 'None', antialiased = True, lw = 1, zorder = 3)

      ax.add_collection(pe)
      ax.add_collection(pf)

      ax.plot(Gaia_table.ra[~Gaia_table.clean_label], Gaia_table.dec[~Gaia_table.clean_label], '.', color = '0.6', ms = 0.75, zorder = 0)
      ax.plot(Gaia_table.ra[Gaia_table.clean_label], Gaia_table.dec[Gaia_table.clean_label], '.', color = '0.2', ms = 0.75, zorder = 1)

      legend_elements = [Line2D([0], [0], marker='.', color='None', markeredgecolor='0.2', markerfacecolor='0.2', label = 'Good stars'), Line2D([0], [0], marker='.', color='None', markeredgecolor='0.6', markerfacecolor='0.6', label = 'Bad stars')]

      if not fields_data.empty:

         for coo, obs_id in fields_data.groupby(['ra','dec']).apply(lambda x: x.index.tolist()).items():
             if len(obs_id) > 1:
                 obs_id = '%i-%i'%(min(obs_id)+1, max(obs_id)+1)
             else:
                 obs_id = obs_id[0]+1
             ax.annotate(obs_id, xy=(coo[0], coo[1]), xycoords='data', color = 'k', zorder = 3)

         for ii, (t_bl, obs_id) in enumerate(fields_data.groupby(['t_baseline']).apply(lambda x: x.index.tolist()).items()):
             if len(obs_id) > 1:
                 obs_id = '%i-%i, %.2f years'%(min(obs_id)+1, max(obs_id)+1, t_bl)
             else:
                 obs_id = '%i, %.2f years'%(obs_id[0]+1, t_bl)
             t = ax.annotate(obs_id, xy=(0.05, 0.95-0.05*ii), xycoords='axes fraction', fontsize = 9, color = 'k', zorder = 3)
             t.set_path_effects([withStroke(foreground="w", linewidth=3)])

         if [0,1,0,0.2] in fields_data.download_color.tolist():
            legend_elements.extend([Patch(facecolor=[0,1,0,0.2], edgecolor='0.4',
                                    label='Previously downloaded'),
                                    Patch(facecolor=[1,1,1,0.2], edgecolor='0.4',
                                    label='Not yet downloaded')])

         for filter, filter_color in fields_data.groupby(['filter'])['filter_color'].first().items():
            legend_elements.append(Patch(facecolor='1', edgecolor=filter_color, label=filter))

      ax.set_xlim(ra_lims[0], ra_lims[1])
      ax.set_ylim(dec_lims[0], dec_lims[1])
      ax.grid()

      ax.set_xlabel(r'RA [$^\circ$]')
      ax.set_ylabel(r'Dec [$^\circ$]')

      if len(bad_patches) > 0:
         legend_elements.append(Patch(facecolor=[1,1,1,0.2], edgecolor='0.4', hatch='/////',
                                 label='Not enough good stars'))

      ax.legend(handles=legend_elements, prop={'size': 7})

      plt.tight_layout()

      plt.savefig(name, bbox_inches='tight')

      with plt.rc_context(rc={'interactive': False}):
         plt.gcf().show()

   obs_table['field_id'] = ['(%i)'%(ii+1) for ii in np.arange(len(obs_table))]
   return obs_table


def search_mast(ra, dec, search_width = 0.25, search_height = 0.25, 
   filters = ['any'], project = ['JWST'], t_exptime_min = 50, t_exptime_max = 2500, 
   date_second_epoch = 57388.5, time_baseline = 3650, jwst_im_type = '_CAL'):
   """
   This routine search for JWST observations in MAST at a given position.
   """
   

   ra1 = ra - search_width / 2 - 0.056 / np.cos(np.deg2rad(dec))
   ra2 = ra + search_width / 2 + 0.056 / np.cos(np.deg2rad(dec))
   dec1 = dec - search_height / 2 - 0.056
   dec2 = dec + search_height / 2 + 0.056

   t_max = date_second_epoch - time_baseline
   
   if type(filters) is not list:
      filters = [filters]
   
   allowed_inst = ['NIRISS', 'NIRISS/IMAGE', 'NIRCAM', 'NIRCAM/IMAGE', 'MIRI', 'MIRI/IMAGE']
   print('filters', filters)
   obs_table = Observations.query_criteria(dataproduct_type=['image'], obs_collection=['JWST'], s_ra=[ra1, ra2], s_dec=[dec1, dec2], instrument_name=allowed_inst, t_max=[0, t_max], project = project)

   #print("obs_table observations.query_criteria", obs_table)
   #allowed_inst = ['ACS/WFC', 'WFC3/UVIS']
   # allowed_inst = ['ACS/WFC', 'ACS/HRC', 'WFC3/UVIS']
   #obs_table = Observations.query_criteria(dataproduct_type=['image'], obs_collection=['HST'], s_ra=[ra1, ra2], s_dec=[dec1, dec2], instrument_name=allowed_inst, t_max=[0, t_max], filters = filters, project = project)
   print(obs_table)
   data_products_by_obs = search_data_products_by_obs(obs_table,jwst_im_type=jwst_im_type)

   #Pandas is easier:
   obs_table = obs_table.to_pandas()
   data_products_by_obs = data_products_by_obs.to_pandas()
   
   #print("data products by obs to pandas (not empty)", data_products_by_obs)
   # We are only interested in FLC and DRZ images (for HST)
   data_products_by_obs = data_products_by_obs.loc[~data_products_by_obs.project.str.contains('HAP'), :]
   #print("data products by obs HAP (not empty)", data_products_by_obs)
   #print("data products by obs loc (not empty)", data_products_by_obs.loc[data_products_by_obs.productSubGroupDescription == jwst_im_type[1:].upper(), :])
   #print("data products by obs groupby (not empty)", data_products_by_obs.loc[data_products_by_obs.productSubGroupDescription == jwst_im_type[1:].upper(), :].groupby(['parent_obsid'])['parent_obsid'].count())
    
   obs_table = obs_table.merge(data_products_by_obs.loc[data_products_by_obs.productSubGroupDescription == jwst_im_type[1:].upper(), :].groupby(['parent_obsid'])['parent_obsid'].count().rename_axis('obsid').rename('n_exp'), on = ['obsid'])

   obs_table['i_exptime'] = obs_table['t_exptime'] / obs_table['n_exp']

   #For convenience we add an extra column with the time baseline
   obs_time = Time(obs_table['t_max'], format='mjd')
   obs_time.format = 'iso'
   obs_time.out_subfmt = 'date'
   obs_table['obs_time'] = obs_time

   obs_table['t_baseline'] = round((date_second_epoch - obs_table['t_max']) / 365.2422, 2)
   print('obs_table[t_max]', obs_table['t_max'])
   print('date_second_epoch', date_second_epoch)
    
   obs_table['filters'] = obs_table['filters'].str.strip('; CLEAR2L CLEAR1L')

   data_products_by_obs = data_products_by_obs.merge(obs_table.loc[:, ['obsid', 'i_exptime', 'filters', 't_baseline', 's_ra', 's_dec']].rename(columns={'obsid':'parent_obsid'}), on = ['parent_obsid'])
   print('time baseline', obs_table["t_baseline"])

   # We select by individual exp time:
   obs_table = obs_table.loc[(obs_table.i_exptime >= t_exptime_min) & (obs_table.i_exptime <= t_exptime_max) & (obs_table.t_baseline >= time_baseline / 365.2422 )]
   print('filters', obs_table.filters)

   data_products_by_obs = data_products_by_obs.loc[(data_products_by_obs.i_exptime >= t_exptime_min) & (data_products_by_obs.i_exptime <= t_exptime_max) & (data_products_by_obs.t_baseline >= time_baseline / 365.2422 )]
   
   #print("obs_table end of function", obs_table)

   return obs_table.astype({'obsid': 'int64'}).reset_index(drop = True), data_products_by_obs.astype({'parent_obsid': 'int64'}).reset_index(drop = True)


def search_data_products_by_obs(obs_table,jwst_im_type='_CAL'):
   """
   This routine search for images in MAST related to the given observations table.
   """

   try:
      data_products_by_obs = Observations.get_product_list(obs_table)
   except Exception:
      from astropy.table import Table as _Table
      return _Table(names=['obsID', 'parent_obsid', 'obs_collection', 'dataproduct_type',
                           'project', 'productFilename', 'productSubGroupDescription',
                           'description', 'type', 'size', 'dataRights', 'calib_level',
                           'filters', 'productType'])
   print("data_products_by_obs in search function", pd.DataFrame(data_products_by_obs['productSubGroupDescription']).value_counts())
   
   return data_products_by_obs[((data_products_by_obs['productSubGroupDescription'] == jwst_im_type[1:].upper()) | (data_products_by_obs['productSubGroupDescription'] == 'RATE')) & (data_products_by_obs['obs_collection'] == 'JWST')]

def download_JWST_images(data_products_by_obs, path = './'):
   """
   This routine downloads the selected JWST images from MAST.
   """
   if hasattr(data_products_by_obs, 'to_pandas'):
      data_products_by_obs = data_products_by_obs.to_pandas()

   cal_only = data_products_by_obs[data_products_by_obs['productSubGroupDescription'] == 'CAL']

   try:
      images = Observations.download_products(Table.from_pandas(cal_only), download_dir=path)
   except:
      images = Observations.download_products(cal_only, download_dir=path)

   return images


def create_dir(path):
   """
   This routine creates directories.
   """

   if not os.path.isdir(path):
      try:
         os.makedirs(path, exist_ok=True)
         print ("Successfully created the directory %s " % path)
      except OSError:  
         print ("Creation of the directory %s failed" % path)


def members_prob(table, clf, vars, errvars, clipping_prob = 3, data_0 = None):
   """
   This routine will find probable members through scoring of a passed model (clf).
   """

   has_vars = table.loc[:, vars].notnull().all(axis = 1)

   data = table.loc[has_vars, vars]
   err = table.loc[has_vars, errvars]

   clustering_data = table.loc[has_vars, 'clustering_data'] == 1

   results = pd.DataFrame(columns = ['logprob', 'member_logprob', 'member_zca'], index = table.index)

   for var in vars:
      results.loc[:, 'w_%s'%var] = np.nan

   if (clustering_data.sum() > 1):

      if data_0 is None:
         data_0 = data.loc[clustering_data, vars].median().values

      data -= data_0

      clf.fit(data.loc[clustering_data, :])
      
      logprob = clf.score_samples(data)
      label_logprob = logprob >= np.nanmedian(logprob[clustering_data])-clipping_prob*np.nanstd(logprob[clustering_data])

      label_wzca = []
      for mean, covariances in zip(clf.means_, clf.covariances_):
         data_c = data-mean

         if clf.covariance_type == 'full':
            cov = covariances
         elif clf.covariance_type == 'diag':
            cov = np.array([[covariances[0],0], [0, covariances[1]]])
         else:
            cov = np.array([[covariances,0], [0, covariances]])

         eigVals, eigVecs = np.linalg.eig(cov)

         diagw = np.diag(1/((eigVals+.1e-6)**0.5)).real.round(5)
         Wzca = np.dot(np.dot(eigVecs, diagw), eigVecs.T)

         wdata = np.dot(data_c, Wzca)
         werr = np.dot(err, Wzca)

         label_wzca.append(((wdata**2).sum(axis =1) <= clipping_prob**2) & ((werr**2).sum(axis =1) <= clipping_prob**2))

      if len(label_wzca) > 1:
         label_wzca = list(map(all, zip(*label_wzca)))
      else:
         label_wzca = label_wzca[0]

      results.loc[has_vars, 'member_logprob'] = label_logprob
      results.loc[has_vars, 'member_zca'] = label_wzca
      results.loc[has_vars, 'logprob'] = logprob

      for var, wdata_col in zip(vars, wdata.T):
         results.loc[has_vars, 'w_%s'%var] = wdata_col
    
   return results


def pm_cleaning_GMM_recursive(table, vars, errvars, clustering_data = None, alt_table = None, data_0 = None, n_components = 1, covariance_type = 'full', clipping_prob = 3, no_plots = True, verbose = True, plot_name = ''):
   """
   This routine iteratively find members using a Gaussian mixture model.
   """
   
   table['real_data'] = True
   
   if clustering_data is not None:
      table['clustering_data'] = clustering_data
   else:
      table['clustering_data'] = 1

   if alt_table is not None:
      alt_table['real_data'] = False
      alt_table['clustering_data'] = 0
      table = pd.concat([table, alt_table], ignore_index = True, sort=True)

   clf = mixture.GaussianMixture(n_components = n_components, covariance_type = covariance_type, means_init = np.zeros((n_components, len(vars))))
   
   if verbose:
      print('')
      print('Finding member stars...')

   convergence = False
   iteration = 0
   while not convergence:
      if verbose & (iteration > 0):
         print("\rIteration %i, %i objects remain."%(iteration, table.clustering_data.sum()))

      clust = table.loc[:, vars+errvars+['clustering_data']]

      if iteration > 3:
         data_0 = None

      fitting = members_prob(clust, clf, vars, errvars, clipping_prob = clipping_prob,  data_0 = data_0)
      
      # If the ZCA detects too few members, we use the logprob.
      if fitting.member_zca.sum() < 10:
         print('WARNING: Not enough members after ZTA whitening. Switching to selection based on logarithmic probability.')
         table['member'] = fitting.member_logprob
      else:
         table['member'] = fitting.member_zca

      table['logprob'] = fitting.logprob
      table['clustering_data'] = (table.clustering_data == 1) & (table.member == 1)  & (table.real_data == 1)

      if (iteration > 999):
         convergence = True
      elif iteration > 0:
         convergence = fitting.equals(previous_fitting)

      previous_fitting = fitting.copy()
      iteration += 1
   
   if no_plots == False:
      plt.close('all')
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10.5, 4.75), dpi=200)
      ax1.plot(table.loc[table.real_data == 1 ,vars[0]], table.loc[table.real_data == 1 , vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax1.scatter(table.loc[table.clustering_data == 1 ,vars[0]], table.loc[table.clustering_data == 1 ,vars[1]], c = fitting.loc[table.clustering_data == 1, 'logprob'], s = 1, zorder = 1)
      ax1.set_xlabel(r'$\mu_{\alpha*}$')
      ax1.set_ylabel(r'$\mu_{\delta}$')
      ax1.grid()

      t = np.linspace(0, 2*np.pi, 100)
      xx = clipping_prob*np.sin(t)
      yy = clipping_prob*np.cos(t)

      ax2.plot(fitting.loc[table.real_data == 1 , 'w_%s'%vars[0]], fitting.loc[table.real_data == 1 , 'w_%s'%vars[1]], 'k.', ms = 0.5, zorder = 0)
      ax2.scatter(fitting.loc[table.clustering_data == 1 , 'w_%s'%vars[0]], fitting.loc[table.clustering_data == 1 , 'w_%s'%vars[1]], c = fitting.loc[table.clustering_data == 1, 'logprob'].values, s = 1, zorder = 1)
      ax2.plot(xx, yy, 'r-', linewidth = 1)

      ax2.set_xlabel(r'$\sigma(\mu_{\alpha*})$')
      ax2.set_ylabel(r'$\sigma(\mu_{\delta})$')
      ax2.grid()

      try:
         margin = 2*(np.nanstd(table.loc[table.real_data == 1 ,vars[0]])+np.nanstd(table.loc[table.real_data == 1 ,vars[1]]))/2
         ax1.set_xlim(np.nanmedian(table.loc[table.real_data == 1 ,vars[0]])-margin, np.nanmedian(table.loc[table.real_data == 1 ,vars[0]])+margin)
         ax1.set_ylim(np.nanmedian(table.loc[table.real_data == 1 ,vars[1]])-margin, np.nanmedian(table.loc[table.real_data == 1 ,vars[1]])+margin)

         ax2.set_xlim(-2*clipping_prob, 2*clipping_prob)
         ax2.set_ylim(-2*clipping_prob, 2*clipping_prob)

      except:
         pass

      plt.subplots_adjust(wspace=0.3, hspace=0.1)

      plt.savefig(plot_name, bbox_inches='tight')
      plt.close(fig)

   if verbose:
      print('')

   if alt_table is not None:
      return table.loc[table.real_data == 1, 'clustering_data'], fitting.loc[table.real_data == 0, 'clustering_data']
   else:
      return table.clustering_data


def remove_file(file_name):
   """
   This routine removes files
   """

   try:
      os.remove(file_name)
   except:
      pass


def applied_pert(XYmqxyrd_filename):
   """
   This routine will read the XYmqxyrd file and find whether the PSF perturbation worked . If file not found then False.
   """

   try:
      print('try')
      f = open(XYmqxyrd_filename, 'r')
      perts = []
      for index, line in enumerate(f):
         if '# CENTRAL PERT PSF' in line:
            for index, line in enumerate(f):
               if len(line[10:-1].split()) > 0:
                  perts.append([float(pert) for pert in line[10:-1].split()])
               else:
                  break
            f.close()
            break
      if len(perts) > 0:
         return np.array(perts).ptp() != 0
      else:
         print('CAUTION: No information about PSF perturbation found in %s!'%XYmqxyrd_filename)
         return True
   except:
      return False


def get_fmin(i_exptime):
   """
   This routine returns the best value for FMIN to be used in jwst1pass_GH execution based on the integration time
   """

   # These are the values for fmin at the given lower_exptime and upper_exptime
   lower_fmin, upper_fmin = 1000, 5000
   lower_exptime, upper_exptime = 50, 500
   f_bounds = 1000,10000 #original values

   lower_fmin, upper_fmin = 100, 5000
   lower_exptime, upper_exptime = 50, 500
   f_bounds = 10,10000 #KM test values

   return min(max((int(lower_fmin + (upper_fmin - lower_fmin) * (i_exptime - lower_exptime) / (upper_exptime - lower_exptime)), f_bounds[0])), f_bounds[1])


def jwst1pass_GH_multiproc_run(args):
   """
   This routine pipes jwst1pass_GH into multiple threads.
   """

   return jwst1pass_GH(*args)


def make_XYmqxyrd(hstpath,image_name, hst_im_type,outname='uvmqxyrdabgcenW'):
   '''
   Take the output XYmqxyrdSTW, and truncate the outputs to create
   a corresponding XYmqxyrd to save time on running the slow step.
   '''
   if outname == 'XYmqxyrdSTW':
      extra_cols = 3
   elif outname == 'XYMqxyrdSTW':
      extra_cols = 3
   elif outname == 'XYmqxyrdSTWc':
      extra_cols = 4
   elif outname == 'XYMqxyrdSTWc':
      extra_cols = 4
   elif outname == 'uvWqxyrdabgcen':
      extra_cols = 6
   elif outname == 'uvWqxyrdcen':
      extra_cols = 3
   elif outname == 'uvmqxyrdcenW':
      extra_cols = 4
   elif outname == 'uvmqxyrdabgcenW':
      extra_cols = 7
   elif outname == 'uvmqxyrdabgcenW':
      extra_cols = 7
   elif outname == 'uvmqxyrdabgcenWA':
      extra_cols = 8

   else:
      extra_cols = 3

    
   fname = f'{hstpath}{image_name}/{image_name}{hst_im_type.lower()}.{outname}'
   #print('fname', fname)
   found = os.path.isfile(fname)
   if not found:
      print(f'SKIPPING {image_name} because of missing {fname}')
      return fname, found, None, None, None
        
   with open(fname,'r') as f:
      lines = f.readlines()

   with open(f'{hstpath}{image_name}/{image_name}{hst_im_type}.XYmqxyrd','w') as f:
      for j,line in enumerate(lines):
         # if line.startswith('#   ARG0007: OUT='):
         #    # print(line)
         #    lines[j] = '#   ARG0007: OUT=XYmqxyrd\n'
         # elif line.startswith('#  FILEOUT:'):
         #    split_line = line.strip().split('/')
         #    fsplit = split_line[-1].split('.')
         #    fsplit[-1] = 'XYmqxyrd'
         #    split_line[-1] = '.'.join(fsplit)
         #    lines[j] = '/'.join(split_line)+'            \n'
         #    # print(lines[j])
         # elif line.startswith('# (09)  S --'):
         #    continue
         # elif line.startswith('# (10)  T --'):
         #    continue
         # elif line.startswith('# (11)  W --'):
         #    continue
         # elif line.startswith('# (12)  c --'):
         #    continue
            
         if ': GDC=' in line:
            instdet,filt = line.split('/')[-1].split('.fits')[0].split('_')[1:]
         if 'rDAT:' in line:
            rdat = float(line.strip().split()[-1])
        
         if line.startswith('#..'):
            split_line = line.split()
            lines[j] = ' '.join(split_line[:-extra_cols])+'  \n'
            char_sizes = np.zeros(len(split_line[:-extra_cols]))
            for ind,entry in enumerate(split_line[:-extra_cols]):
               char_sizes[ind] = len(entry)
            cut_ind = int(np.sum(char_sizes)+len(char_sizes)-1)
            # print(cut_ind)
         elif not line.startswith('#'):
            #then into the numbers, so lop off the last 3 entries
            if len(line) == 0:
               continue
            elif len(line.split()) == 0:
               continue
            # Robustly capture exactly 8 columns while preserving internal fixed-width spacing
            # Use regex to match 8 non-whitespace blocks and their preceding/internal whitespace
            match = re.match(r'^(\s*(?:\S+\s+){7}\S+)', line)
            if match:
               new_line = match.group(1).ljust(100) # Pad to maintain legacy line length if possible
            else:
               try:
                  new_line = line[:cut_ind]
               except:
                  raise ValueError(f'BAD ENTRY in {fname} on line {j}: {line}')

            split_new_line = new_line.split()
            if len(split_new_line) == 0:
               continue
            elif len(split_new_line) != 8:
               raise ValueError(f'BAD ENTRY in {fname} on line {j}: {new_line}')
            lines[j] = new_line+'\n'
        
         f.write(lines[j])
   return fname,found,rdat,instdet,filt

def availabe_PSFs(psf_path):
   PSFs = {}
   for camera in os.listdir(psf_path):
      if not os.path.isdir(f'{psf_path}{camera}/'):
         continue
      if camera == '.DS_Store':
         continue
      PSFs[camera] = []
      for fname in os.listdir(f'{psf_path}{camera}/'):
         if fname.endswith('.fits'):
            split_f = fname.split('_')
            if split_f[-1].startswith('SM'):
               curr_filter = split_f[-2]#+'_'+split_f[-1][:-5]
            else:
               curr_filter = split_f[-1][:-5]
            if curr_filter not in PSFs[camera]:
               PSFs[camera].append(curr_filter)
   #add 'P', because we have .replace('P', '') in previous lines
   #add 'C', because there are e.g. F1065C
   poss_filt_ends = ['LP','L','W','M','N', 'P', 'C']

   allowed_psfs = {}
   psf_filt_waves = {}

   for camera in PSFs:
      allowed_psfs[camera] = {'all':PSFs[camera]}
      psf_filt_waves[camera] = {'filters':np.array(PSFs[camera]),
                                 'waves':np.zeros(len(PSFs[camera]))}

      for j,filt in enumerate(PSFs[camera]):
         curr_wave = filt.replace(r'F', '')
         for filt_end in poss_filt_ends:
            curr_wave = curr_wave.replace(filt_end,'')
         psf_filt_waves[camera]['waves'][j] = int(curr_wave)

      for end in poss_filt_ends:
         allowed_psfs[camera][end] = []
         for filt in PSFs[camera]:
            if filt.endswith(end):
               allowed_psfs[camera][end].append(filt)

   return allowed_psfs,psf_filt_waves


def availabe_GDCs(gdc_path):
   GDCs = {}
   for camera in os.listdir(gdc_path):
      if not os.path.isdir(f'{gdc_path}{camera}/'):
         continue
      if camera == '.DS_Store':
         continue
      GDCs[camera] = []
      for fname in os.listdir(f'{gdc_path}{camera}/'):
         if fname.endswith('.fits'):
            GDCs[camera].append(fname.split('_')[-1][:-5])

   poss_filt_ends = ['LP','L','W','M','N']
   gdc_filt_waves = {}
   allowed_gdcs = {}
   for camera in GDCs:
      allowed_gdcs[camera] = {'all':GDCs[camera]}
      gdc_filt_waves[camera] = {'filters':np.array(GDCs[camera]),
                                 'waves':np.zeros(len(GDCs[camera]))}
      for j,filt in enumerate(GDCs[camera]):
         curr_wave = filt.replace(r'F', '')
         for filt_end in poss_filt_ends:
            curr_wave = curr_wave.replace(filt_end,'')
         gdc_filt_waves[camera]['waves'][j] = int(curr_wave)
      for end in poss_filt_ends:
         allowed_gdcs[camera][end] = []
         for filt in GDCs[camera]:
            if filt.endswith(end):
               allowed_gdcs[camera][end].append(filt)

   return allowed_gdcs,gdc_filt_waves

# def xryr2xcyc(xy_raw,stdgdc_fname):
#    '''
#    Takes the xy coordinates in raw pixels, and then corrects 
#    for geometric distortions
#    '''
#    stdgdc_data = fits.open(stdgdc_fname)
    
#    XGC = stdgdc_data[1].data.T
#    YGC = stdgdc_data[2].data.T

def xryr2xcyc(xy_raw,XGC,YGC):
   '''
   Takes the xy coordinates in raw pixels, and then corrects 
   for geometric distortions.
   Coordinates are 1-based.
   '''
   # Robustly handle coordinates outside the GDC grid boundaries
   nx, ny = XGC.shape
   xy = xy_raw.copy()

   # Clip coordinates to be within [1.0, nx - 0.001] and [1.0, ny - 0.001]
   # This ensures ipix-1 >= 0 and ipix < nx/ny
   xy[:,0] = np.clip(xy[:,0], 1.0, nx - 0.001)
   xy[:,1] = np.clip(xy[:,1], 1.0, ny - 0.001)

   ipix = xy[:,0].astype(int) 
   jpix = xy[:,1].astype(int) 
   fx = xy[:,0]-ipix
   fy = xy[:,1]-jpix
   xc = (1-fx)*(1-fy)*XGC[ipix-1,jpix-1]/1.0\
        +(1-fx)*fy*XGC[ipix-1,jpix-1+1]/1.0\
        +fx*(1-fy)*XGC[ipix-1+1,jpix-1]\
        +fx*fy*XGC[ipix-1+1,jpix-1+1]/1.0
   yc = (1-fx)*(1-fy)*YGC[ipix-1,jpix-1]/1.0\
        +(1-fx)*fy*YGC[ipix-1,jpix-1+1]/1.0\
        +fx*(1-fy)*YGC[ipix-1+1,jpix-1]/1.0\
        +fx*fy*YGC[ipix-1+1,jpix-1+1]/1.0 

   xy_c = np.array([xc,yc]).T

   return xy_c

def xryr2xcyc_derivs(xy_raw,XGC,YGC):
   '''
   Takes the xy coordinates in raw pixels, and then measures the 
   derivative of the geometric distortion correction
   '''
   offset = 0.001
   xy_raw_dx_p = xy_raw+np.array([[offset,0]])
   xy_raw_dx_m = xy_raw+np.array([[-offset,0]])
   xy_raw_dy_p = xy_raw+np.array([[0,offset]])
   xy_raw_dy_m = xy_raw+np.array([[0,-offset]])

   xy_raw_sets = np.array([xy_raw_dx_p, xy_raw_dx_m, xy_raw_dy_p, xy_raw_dy_m])
   new_xys = np.zeros((4,len(xy_raw),2))

   for j, xy_set in enumerate(xy_raw_sets):
       new_xys[j] = xryr2xcyc(xy_set, XGC, YGC)
   dxc_dxr = (new_xys[0,:,0]-new_xys[1,:,0])/(xy_raw_sets[0,:,0]-xy_raw_sets[1,:,0])
   dxc_dyr = (new_xys[2,:,0]-new_xys[3,:,0])/(xy_raw_sets[2,:,1]-xy_raw_sets[3,:,1])
   dyc_dxr = (new_xys[0,:,1]-new_xys[1,:,1])/(xy_raw_sets[0,:,0]-xy_raw_sets[1,:,0])
   dyc_dyr = (new_xys[2,:,1]-new_xys[3,:,1])/(xy_raw_sets[2,:,1]-xy_raw_sets[3,:,1])

   dxyc_dxyr = np.zeros((len(xy_raw),2,2))
   dxyc_dxyr[:,0,0] = dxc_dxr
   dxyc_dxyr[:,0,1] = dxc_dyr
   dxyc_dxyr[:,1,0] = dyc_dxr
   dxyc_dxyr[:,1,1] = dyc_dyr

   return dxyc_dxyr


def jwst1pass_GH(JWST_path, exec_path, obs_id, JWST_image, force_fmin, 
               force_jwst1pass, verbose, auto_fmin_mult, jwst_im_type,
               print_only=False, psf_pert=False, python_jwst1pass=False):
   """
   This routine will execute jwst1pass_GH Fortran routine using the correct arguments.
   """

   outname = 'uvmqxyrdabgcenW'
   output_dir = JWST_path+'mastDownload/JWST/'+obs_id+'/'
   #JWST_path = './LMC/JWST/'
   #print('obs_id', obs_id)
   JWST_image_filename = output_dir+JWST_image
   #print('JWST_image_filename', os.path.isfile(JWST_image_filename))
   XYmqxyrd_filename = JWST_image_filename.split('.fits')[0]+'.%s'%outname
   XYmqxyrd_filename_other = JWST_image_filename.split('.fits')[0]+'.XYmqxyrd'
   #print('XYmqxyrd_filename', os.path.isfile(XYmqxyrd_filename))
   #print('XYmqxyrd_filename_other', os.path.isfile(XYmqxyrd_filename_other))

   # if not os.path.isfile(XYmqxyrd_filename):
   #    return

   ref_coords = []
   art_fname = JWST_image_filename.split('.fits')[0]+'_ART.xym'

   if not print_only:
      if force_jwst1pass or (force_fmin is not None):
         remove_file(XYmqxyrd_filename)
         remove_file(XYmqxyrd_filename_other)
         remove_file(art_fname)
            
   good_image = False
   if os.path.isfile(XYmqxyrd_filename):
      good_image = True
        
   if not os.path.isfile(XYmqxyrd_filename):
      print('this if statement is executed', XYmqxyrd_filename)
      if verbose:
         print('Finding sources in', JWST_image)
      
      if not os.path.isfile(JWST_image_filename):
         print(f'RSKIPPING image {JWST_image} because no JWST image was found')
         return ''

      if python_jwst1pass:
         os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + os.path.abspath(os.path.join(exec_path, 'jwst1pass_py_v2'))

      #Read information from the header of the image
      hdul = fits.open(JWST_image_filename)
      instrument = hdul[0].header['INSTRUME']
      detector = hdul[0].header['DETECTOR']
        
      for hdr_ind in range(len(hdul)):
         if ('CRPIX1' in hdul[hdr_ind].header):
            x_ref,y_ref = hdul[hdr_ind].header['CRPIX1'],hdul[hdr_ind].header['CRPIX2']
            ra_ref,dec_ref = hdul[hdr_ind].header['CRVAL1'],hdul[hdr_ind].header['CRVAL2']
            curr_coords = [x_ref,y_ref]
            if curr_coords not in ref_coords:
               ref_coords.append(curr_coords)
            # print(hdr_ind,hdul[0].header['INSTRUME'],hdul[0].header['DETECTOR'],hdul[hdr_ind].header['CCDCHIP'],x,y,ra,dec)
      for test_coord in [[1024.5,1024.5]]:
         if test_coord not in ref_coords:
            ref_coords.append(curr_coords)
      ref_coords = np.array(ref_coords)
        
      #print("hdul", hdul)

      # filter = hdul[0].header['PUPIL']
      # try:
      #    filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
      # except:
      if 'FILTER' in hdul[0].header:
         filter = hdul[0].header['FILTER']
         if 'CLEAR' in filter:
            filter = hdul[0].header['PUPIL']
      else:
         filter = hdul[0].header['PUPIL']
      
      #add 'P' since this ending appears in previous lines... .replace('P', '')
      #print("hdul filter", filter, JWST_image_filename)
      poss_filt_ends = ['LP','L','W','M','N', 'P']
      for filt_end in poss_filt_ends:
         if filter.endswith(filt_end):
            break
      curr_filt_end = filt_end

      #check to see if that filter is availabe in PSF,
      #and if not, grabs the closest

      # gdc_path = f'{exec_path}/lib/STDGDCs/'
      # allowed_gdcs = availabe_GDCs(gdc_path)[f'{instrument}{detector}']

      psf_path = f'{exec_path}/lib/STDPSFs/'
      allowed_psfs,psf_filt_waves = availabe_PSFs(psf_path)
      #print('allowed_psfs', allowed_psfs)
      allowed_psfs = allowed_psfs[f'{instrument}']
      #allowed_psfs = allowed_psfs[f'{instrument}{detector}']
      #psf_filt_waves = psf_filt_waves[f'{instrument}{detector}']
      psf_filt_waves = psf_filt_waves[f'{instrument}']
      if filter in allowed_psfs['all']:
         filter_psf = filter
      else:
         curr_wave = int(filter.replace(r'F', '').replace(curr_filt_end,''))
         wave_diffs = np.abs(curr_wave-psf_filt_waves['waves'])
         best_diff = np.min(wave_diffs)
         best_match_inds = np.where(wave_diffs== best_diff)[0]
         best_match_filts = psf_filt_waves['filters'][best_match_inds]
         if len(best_match_inds) == 0:
            best_match_ind = best_match_inds[0]
            filter_psf = psf_filt_waves['filters'][best_match_ind]
         else:
            #maybe change in the future to consider which order of
            #filters (L,W,M,N) to check in order
            best_match_ind = best_match_inds[0]
            filter_psf = psf_filt_waves['filters'][best_match_ind]

      # # No standard PSF library for F555W filter. We use the F606W.
      # if ('F555W' in filter) and (instrument == 'ACS') and (detector == 'WFC'):
      #    filter_psf = filter.replace('F555W', 'F606W')
      # else:
      #    filter_psf = filter

      t_max = hdul[0].header['EXPEND']

      if force_fmin is None:
         exptime = hdul[0].header['EFFINTTM']
         fmin = get_fmin(exptime)*auto_fmin_mult
         # if fmin == 10000:
         #    print('Skipping %s. because of large integration time, %.2f sec'%(HST_image,exptime))
         #    return
         # if exptime > 1300:
         # if exptime > 1500:
         if exptime > 1800:
            print('Skipping %s because of large integration time (%.2f sec).'%(JWST_image,exptime))
            return ''
      else:
         fmin = force_fmin

      if verbose:
         print('%s exptime = %.1f. Using fmin = %i'%(JWST_image, hdul[0].header['EFFINTTM'], fmin))

      #psf_filename = '%s/lib/STDPSFs/%s%s/STDPSF_%s%s_%s.fits'%(exec_path, instrument, detector, instrument, detector, filter_psf)
      psf_filename = '%s/lib/STDPSFs/%s/STDPSF_%s_%s.fits'%(exec_path, instrument, instrument, filter_psf)
      #gdc_filename = '%s/lib/STDGDCs/%s%s/STDGDC_%s%s_%s.fits'%(exec_path, instrument, detector, instrument, detector, filter)
      gdc_filename = '%s/lib/STDGDCs/%s/STDGDC_%s_%s.fits'%(exec_path, instrument, instrument, filter)
      print('psf_filename', psf_filename)
   if (not os.path.isfile(XYmqxyrd_filename)) or (not os.path.isfile(art_fname)):
      if (os.path.isfile(psf_filename) or print_only) and os.path.isfile(gdc_filename):
         n_repeat = 0
         if psf_pert:
            pert_grid = 5
            while not applied_pert(XYmqxyrd_filename) and (pert_grid > 0):
               # bashCommand = "%s/fortran_codes/hst1pass_GH.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=XYmqxyrd OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, output_dir, HST_image_filename)
               # bashCommand = "%s/fortran_codes/hst1pass_GH_errs.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, output_dir, HST_image_filename)


               #bashCommand = "%s/fortran_codes/jwst1pass_GH_errs.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, output_dir, JWST_image_filename)
               if not python_jwst1pass:
                  bashCommand = "%s/fortran_codes/jwst1pass_LC.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, JWST_image_filename)
                  my_env = os.environ.copy()
               else:
                  python_script = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2/scripts/jwst1pass_run.py"))
                  lib_dir = os.path.abspath(os.path.join(exec_path, "lib"))
                  fortran_out_prefix = os.path.abspath(os.path.join(output_dir, JWST_image.replace(".fits", "")))
                  # Include --plot and use absolute paths
                  bashCommand = f"python3 {python_script} --image {os.path.abspath(JWST_image_filename)} --lib_dir {lib_dir} --fmin {fmin} --hmin 4 --verbose --plot --fortran_output {fortran_out_prefix}.{outname}"
                  
                  # Set up environment with PYTHONPATH
                  my_env = os.environ.copy()
                  jwst1pass_pkg_path = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2"))
                  my_env["PYTHONPATH"] = jwst1pass_pkg_path + ":" + my_env.get("PYTHONPATH", "")
               
               print('while if', bashCommand) 
               #move file
               curr_file = JWST_image.replace('.fits',f'.{outname}')
               if not python_jwst1pass:
                  try:
                     os.rename(f'./{curr_file}',f'{output_dir}/{curr_file}')
                  except:
                     pass
               print('curr_file path', os.path.abspath(curr_file))
               
               if print_only:
                  print(bashCommand)
                  break
               else:
                  # Run with retry logic for shared region issues
                  repeat_err_msg = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
                  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
                  output, error = process.communicate()
                  
                  while (repeat_err_msg in error):
                     process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
                     output, error = process.communicate()

                  pert_grid -= 1
         else:
            pert_grid = 0
            # bashCommand = "%s/fortran_codes/hst1pass_GH_errs.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, output_dir, HST_image_filename)
            
            #bashCommand = "%s/fortran_codes/jwst1pass_JA.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, JWST_image_filename)
            #bashCommand = "%s/fortran_codes/jwst1pass.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, JWST_image_filename)
            if not python_jwst1pass:
               bashCommand = "%s/fortran_codes/jwst1pass_LC.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, JWST_image_filename)
               my_env = os.environ.copy()
            else:
               python_script = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2/scripts/jwst1pass_run.py"))
               lib_dir = os.path.abspath(os.path.join(exec_path, "lib"))
               fortran_out_prefix = os.path.abspath(os.path.join(output_dir, JWST_image.replace(".fits", "")))
               # Include --plot and use absolute paths
               bashCommand = f"python3 {python_script} --image {os.path.abspath(JWST_image_filename)} --lib_dir {lib_dir} --fmin {fmin} --hmin 4 --verbose --plot --fortran_output {fortran_out_prefix}.{outname}"
               
               # Set up environment with PYTHONPATH
               my_env = os.environ.copy()
               jwst1pass_pkg_path = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2"))
               my_env["PYTHONPATH"] = jwst1pass_pkg_path + ":" + my_env.get("PYTHONPATH", "")
            
            print('out name + JWST_image_filename', outname, JWST_image_filename)

            print(bashCommand)
            
            #print('bashCommand', bashCommand)
            if print_only:
               print(bashCommand)
            else:
               repeat_err_msg = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
               process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
               output, error = process.communicate()
               
               while (repeat_err_msg in error):
                  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
                  output, error = process.communicate()
                  n_repeat += 1

            #if using the hst1pass_JA.e version, move the file to the correct location
            #move file to another place
            curr_file = JWST_image.replace('.fits',f'.{outname}')
            print('curr_file')
            #if using the hst1pass_JA.e version, move the file to the correct location
            #move file to another place
            curr_file = JWST_image.replace('.fits',f'.{outname}')
            print('curr_file')
            if not python_jwst1pass:
               try:
                  os.rename(f'./{curr_file}',f'{output_dir}/{curr_file}')
               except:
                  pass
            print('curr_file path1', os.path.abspath(curr_file))
            # try:
            #    shutil.move(os.path.abspath(curr_file), output_dir)
            # except FileNotFoundError:
            #    print(f"Error: Source file '{source_file}' not found.")
            # print('curr_file path2', os.path.abspath(curr_file))
            # sdnvk
            if os.path.isfile(f'{output_dir}/{curr_file}'):
               #make synthetic stars to measure position uncertainties
               outdata = np.genfromtxt(f'{output_dir}/{curr_file}')
               if (len(outdata) > 0):
                  if len(outdata.shape) == 1:
                     outdata = outdata[None,:]
                  out_xy_orig = outdata[:,[4,5]]
                  out_mag_orig = outdata[:,2]
                  out_q_orig = outdata[:,3]
                  good_q = out_q_orig > 0
                  if np.sum(good_q) > 0:
                     good_mag_range = np.nanpercentile(out_mag_orig[out_q_orig > 0],[0,100])
                  else:
                     good_mag_range = np.nanpercentile(out_mag_orig,[0,100])
                  good_mag_range[0] -= 0.01
                  good_mag_range[1] += 0.01
                    
                  #pix_dims = (4096,4096)
                  pix_dims = (2048,2048)
                  n_x = 11
                  n_y = 11
                  n_mags = 15
                  min_peak_spacing = 10 #pixels

                  mags = np.linspace(good_mag_range[0],good_mag_range[1],n_mags)

                  x_poses = np.linspace(0,pix_dims[0],n_x+1)
                  y_poses = np.linspace(0,pix_dims[1],n_y+1)

                  x_poses = 0.5*(x_poses[1:]+x_poses[:-1])
                  y_poses = 0.5*(y_poses[1:]+y_poses[:-1])

                  xx,yy,mm = np.meshgrid(x_poses,y_poses,mags)
                  x_offsets = ((np.arange(0,n_mags)-n_mags//2)*min_peak_spacing)
                  xx += x_offsets[None,None,:]
                  yy += x_offsets[None,None,:]

                  xx = np.ravel(xx)
                  yy = np.ravel(yy)
                  mm = np.ravel(mm)
         
                  with open(art_fname,'w') as f:
                     f.write('#....x.... ....y.... ....m....\n')
                     for j in range(len(xx)):
                        f.write(f'{xx[j]:9.4f} {yy[j]:9.4f} {mm[j]:9.4f}\n')
                        
                  art_command = 'ART_xym=%s'%(art_fname)
                  if not python_jwst1pass:
                     new_bashCommand = '%s/fortran_codes/jwst1pass_LC.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s %s %s'%(exec_path, fmin, gdc_filename, psf_filename, outname, JWST_image_filename,art_command)
                  else:
                     python_script = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2/scripts/jwst1pass_run.py"))
                     lib_dir = os.path.abspath(os.path.join(exec_path, "lib"))
                     fortran_out_prefix = os.path.abspath(os.path.join(output_dir, JWST_image.replace(".fits", "")))
                     new_bashCommand = f"python3 {python_script} --image {os.path.abspath(JWST_image_filename)} --lib_dir {lib_dir} --fmin {fmin} --hmin 4 --verbose --plot --input_catalog {os.path.abspath(art_fname)} --fortran_output {fortran_out_prefix}_ART.{outname}"
                     
                     # Set up environment with PYTHONPATH
                     my_env = os.environ.copy()
                     jwst1pass_pkg_path = os.path.abspath(os.path.join(exec_path, "jwst1pass_py_v2"))
                     my_env["PYTHONPATH"] = jwst1pass_pkg_path + ":" + my_env.get("PYTHONPATH", "")
                  
                  print('new_bashcommand', new_bashCommand)
                  
                  repeat_err_msg = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
                  error = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
                  if print_only:
                     print(new_bashCommand)
                  else:
                     while (repeat_err_msg in error):
                        if python_jwst1pass:
                           process = subprocess.Popen(new_bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
                        else:
                           process = subprocess.Popen(new_bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        output, error = process.communicate()
                        # n_repeat += 1
                  #    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                  #    output, error = process.communicate()
                  #if using the hst1pass_JA.e version, move the file to the correct location
                  if not python_jwst1pass:
                     new_fname = curr_file.replace(f".{outname}",f"_ART.{outname}")
                     #print('curr_file', curr_file, 'new_fname', new_fname)
                     os.rename(f'./{curr_file}',f'{output_dir}{new_fname}')
                     
                     new_fname = curr_file.replace(f"_cal.{outname}",f"_art.xymnfo")
                     new_adjust_fname = curr_file.replace(f"_cal.{outname}",f"_cal_art.xymnfo")
                     try:
                        os.rename(f'{output_dir}{new_fname}',f'{output_dir}{new_adjust_fname}')
                     except:
                        pass
                  #os.rename(f'./{new_fname}',f'{output_dir}/{new_adjust_fname}')
                  
                  

         if verbose:
            print('%s PERT PSF = %sx%s, Repeated %s times'%(JWST_image, pert_grid, pert_grid, n_repeat))

         remove_file(JWST_image.replace(jwst_im_type,'_psf'))
      else:

         if not os.path.isfile(psf_filename):
            print('WARNING: %s file not found!'%os.path.basename(psf_filename))
         if not os.path.isfile(gdc_filename):
            print('WARNING: %s file not found!'%os.path.basename(gdc_filename))
         print('Skipping %s.'%JWST_image)


   image_name = obs_id
   jwstpath = JWST_path+'mastDownload/JWST/'

   outfname,found_outfile,rdat,instdet,filt = make_XYmqxyrd(jwstpath,image_name,jwst_im_type,outname=outname)
   if found_outfile:
        outdata = np.genfromtxt(outfname)
        if (len(outdata) > 0):
            if len(outdata.shape) == 1:
                outdata = outdata[None,:]
            out_xy_orig = outdata[:,[4,5]]
    
            stdgdc_data = fits.open(gdc_filename)
            XGC = stdgdc_data[1].data.T
            YGC = stdgdc_data[2].data.T
            stdgdc_data.close()
        
            # corr_vals = xryr2xcyc(out_xy_orig,gdc_filename)
            # corr_derivs = xryr2xcyc_derivs(out_xy_orig,gdc_filename)
            corr_vals = xryr2xcyc(out_xy_orig,XGC,YGC)
            corr_derivs = xryr2xcyc_derivs(out_xy_orig,XGC,YGC)
            pd.DataFrame({
                'XC':corr_vals[:,0],\
                'YC':corr_vals[:,1],\
                'dXC_dXR':corr_derivs[:,0,0],\
                'dXC_dYR':corr_derivs[:,0,1],\
                'dYC_dXR':corr_derivs[:,1,0],\
                'dYC_dYR':corr_derivs[:,1,1],\
            }).to_csv(JWST_image_filename.split('.fits')[0]+'_GDC_vals.csv')
            
            orig_vals = ref_coords
        corr_vals = xryr2xcyc(orig_vals,XGC,YGC)
        corr_derivs = xryr2xcyc_derivs(orig_vals,XGC,YGC)
        pd.DataFrame({
             'XC':corr_vals[:,0],\
             'YC':corr_vals[:,1],\
             'dXC_dXR':corr_derivs[:,0,0],\
             'dXC_dYR':corr_derivs[:,0,1],\
             'dYC_dXR':corr_derivs[:,1,0],\
             'dYC_dYR':corr_derivs[:,1,1],\
             'XR':orig_vals[:,0],\
             'YR':orig_vals[:,1],\
         }).to_csv(JWST_image_filename.split('.fits')[0]+'_GDC_vals_REFCOORDS.csv')
            
   
   # try:
   #    bashCommand
   # except:
   #    print('BAD IMAGE NAME:',image_name)

   # return bashCommand

   try:
      bashCommand
      return bashCommand
   except:
      if not good_image:
         print('BAD IMAGE NAME:',image_name)
      return ''

   XYmqxyrd_filename = JWST_image_filename.split('.fits')[0]+'.XYmqxyrd'
   print('XYmqxyrd_filename', os.path.isfile(XYmqxyrd_filename), XYmqxyrd_filename)
   #print('XYmqxyrd_filename_other', os.path.isfile(XYmqxyrd_filename_other))

   # if not os.path.isfile(XYmqxyrd_filename):
   #    return

   if not print_only:
      if force_jwst1pass or (force_fmin is not None):
         remove_file(XYmqxyrd_filename)
         remove_file(XYmqxyrd_filename_other)
         remove_file(art_fname)
            
   good_image = False
   if os.path.isfile(XYmqxyrd_filename):
      good_image = True
        
   if not os.path.isfile(XYmqxyrd_filename):
      print('RRthis if statement is executed', XYmqxyrd_filename)
      if verbose:
         print('Finding sources in', JWST_image)
      
      if not os.path.isfile(JWST_image_filename):
         print(f'RSKIPPING image {JWST_image} because no JWST image was found')
         return ''

      if python_jwst1pass:
         os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + os.path.abspath(os.path.join(exec_path, 'jwst1pass_py_v2'))

      #Read information from the header of the image
      hdul = fits.open(JWST_image_filename)
      instrument = hdul[0].header['INSTRUME']
      detector = hdul[0].header['DETECTOR']
        
      for hdr_ind in range(len(hdul)):
         if ('CRPIX1' in hdul[hdr_ind].header):
            x_ref,y_ref = hdul[hdr_ind].header['CRPIX1'],hdul[hdr_ind].header['CRPIX2']
            ra_ref,dec_ref = hdul[hdr_ind].header['CRVAL1'],hdul[hdr_ind].header['CRVAL2']
            curr_coords = [x_ref,y_ref]
            if curr_coords not in ref_coords:
               ref_coords.append(curr_coords)
            # print(hdr_ind,hdul[0].header['INSTRUME'],hdul[0].header['DETECTOR'],hdul[hdr_ind].header['CCDCHIP'],x,y,ra,dec)
      for test_coord in [[1024.5,1024.5]]:
         if test_coord not in ref_coords:
            ref_coords.append(curr_coords)
      ref_coords = np.array(ref_coords)
        
      #print("hdul", hdul)

      # filter = hdul[0].header['PUPIL']
      # try:
      #    filter = [filter for filter in [hdul[0].header['FILTER1'], hdul[0].header['FILTER2']] if 'CLEAR' not in filter][0]
      # except:
      if 'FILTER' in hdul[0].header:
         filter = hdul[0].header['FILTER']
         if 'CLEAR' in filter:
            filter = hdul[0].header['PUPIL']
      else:
         filter = hdul[0].header['PUPIL']
      
      #add 'P' since this ending appears in previous lines... .replace('P', '')
      #print("hdul filter", filter, JWST_image_filename)
      poss_filt_ends = ['LP','L','W','M','N', 'P']
      for filt_end in poss_filt_ends:
         if filter.endswith(filt_end):
            break
      curr_filt_end = filt_end

      #check to see if that filter is availabe in PSF,
      #and if not, grabs the closest

      # gdc_path = f'{exec_path}/lib/STDGDCs/'
      # allowed_gdcs = availabe_GDCs(gdc_path)[f'{instrument}{detector}']

      psf_path = f'{exec_path}/lib/STDPSFs/'
      allowed_psfs,psf_filt_waves = availabe_PSFs(psf_path)
      #print('allowed_psfs', allowed_psfs)
      allowed_psfs = allowed_psfs[f'{instrument}']
      #allowed_psfs = allowed_psfs[f'{instrument}{detector}']
      #psf_filt_waves = psf_filt_waves[f'{instrument}{detector}']
      psf_filt_waves = psf_filt_waves[f'{instrument}']
      if filter in allowed_psfs['all']:
         filter_psf = filter
      else:
         curr_wave = int(filter.replace(r'F', '').replace(curr_filt_end,''))
         wave_diffs = np.abs(curr_wave-psf_filt_waves['waves'])
         best_diff = np.min(wave_diffs)
         best_match_inds = np.where(wave_diffs== best_diff)[0]
         best_match_filts = psf_filt_waves['filters'][best_match_inds]
         if len(best_match_inds) == 0:
            best_match_ind = best_match_inds[0]
            filter_psf = psf_filt_waves['filters'][best_match_ind]
         else:
            #maybe change in the future to consider which order of
            #filters (L,W,M,N) to check in order
            best_match_ind = best_match_inds[0]
            filter_psf = psf_filt_waves['filters'][best_match_ind]

      # # No standard PSF library for F555W filter. We use the F606W.
      # if ('F555W' in filter) and (instrument == 'ACS') and (detector == 'WFC'):
      #    filter_psf = filter.replace('F555W', 'F606W')
      # else:
      #    filter_psf = filter

      t_max = hdul[0].header['EXPEND']

      if force_fmin is None:
         exptime = hdul[0].header['EFFINTTM']
         fmin = get_fmin(exptime)*auto_fmin_mult
         # if fmin == 10000:
         #    print('Skipping %s. because of large integration time, %.2f sec'%(HST_image,exptime))
         #    return
         # if exptime > 1300:
         # if exptime > 1500:
         if exptime > 1800:
            print('Skipping %s because of large integration time (%.2f sec).'%(JWST_image,exptime))
            return ''
      else:
         fmin = force_fmin

      if verbose:
         print('%s exptime = %.1f. Using fmin = %i'%(JWST_image, hdul[0].header['EFFINTTM'], fmin))

      #psf_filename = '%s/lib/STDPSFs/%s%s/STDPSF_%s%s_%s.fits'%(exec_path, instrument, detector, instrument, detector, filter_psf)
      psf_filename = '%s/lib/STDPSFs/%s/STDPSF_%s_%s.fits'%(exec_path, instrument, instrument, filter_psf)
      #gdc_filename = '%s/lib/STDGDCs/%s%s/STDGDC_%s%s_%s.fits'%(exec_path, instrument, detector, instrument, detector, filter)
      gdc_filename = '%s/lib/STDGDCs/%s/STDGDC_%s_%s.fits'%(exec_path, instrument, instrument, filter)
      print('psf_filename', psf_filename)
   #if (not os.path.isfile(XYmqxyrd_filename)) or (not os.path.isfile(art_fname)):
      if (os.path.isfile(psf_filename) or print_only) and os.path.isfile(gdc_filename):
         n_repeat = 0
         if psf_pert:
            pert_grid = 5
            while not applied_pert(XYmqxyrd_filename) and (pert_grid > 0):
               # bashCommand = "%s/fortran_codes/hst1pass_GH.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=XYmqxyrd OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, output_dir, HST_image_filename)
               # bashCommand = "%s/fortran_codes/hst1pass_GH_errs.e HMIN=5 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, output_dir, HST_image_filename)


               #bashCommand = "%s/fortran_codes/jwst1pass_GH_errs.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, output_dir, JWST_image_filename)
               if not python_jwst1pass:
                  bashCommand = "%s/fortran_codes/jwst1pass_LC.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s PERT%i=AUTO OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, pert_grid, outname, JWST_image_filename)
               else:
                  python_script = os.path.join(exec_path, "jwst1pass_py_v2/scripts/jwst1pass_run.py")
                  lib_dir = os.path.join(exec_path, "lib")
                  fortran_out_prefix = os.path.join(output_dir, JWST_image.replace(".fits", ""))
                  bashCommand = f"python3 {python_script} --image {JWST_image_filename} --lib_dir {lib_dir} --fmin {fmin} --hmin 4 --verbose --fortran_output {fortran_out_prefix}.{outname}"
               print('while if', bashCommand) 
               #move file to the destination
               #move file
               curr_file = JWST_image.replace('.fits',f'.{outname}')
               if not python_jwst1pass:
                  try:
                     os.rename(f'./{curr_file}',f'{output_dir}/{curr_file}')
                  except:
                     pass
               print('curr_file path', os.path.abspath(curr_file))
               
               try:
                  shutil.move(JWST_image_filename, output_dir.split('j')[0])
               except FileNotFoundError:
                  print(f"Error: Source file '{source_file}' not found.")
                    
               if print_only:
                  print(bashCommand)
                  break
               else:
                  process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                  output, error = process.communicate()
                  pert_grid -= 1
         else:
            pert_grid = 0
            # bashCommand = "%s/fortran_codes/hst1pass_GH_errs.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s OUTDIR=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, output_dir, HST_image_filename)
            
            #bashCommand = "%s/fortran_codes/jwst1pass_JA.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, JWST_image_filename)
            #bashCommand = "%s/fortran_codes/jwst1pass.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, JWST_image_filename)
            if not python_jwst1pass:
               bashCommand = "%s/fortran_codes/jwst1pass_LC.e HMIN=4 FMIN=%s PMAX=999999 GDC=%s PSF=%s OUT=%s %s"%(exec_path, fmin, gdc_filename, psf_filename, outname, JWST_image_filename)
            else:
               python_script = os.path.join(exec_path, "jwst1pass_py_v2/scripts/jwst1pass_run.py")
               lib_dir = os.path.join(exec_path, "lib")
               fortran_out_prefix = os.path.join(output_dir, JWST_image.replace(".fits", ""))
               bashCommand = f"python3 {python_script} --image {JWST_image_filename} --lib_dir {lib_dir} --fmin {fmin} --hmin 4 --verbose --fortran_output {fortran_out_prefix}.{outname}"
            print('second jwst1pass_GH', outname, JWST_image_filename)
            print(bashCommand)
            
            #print('bashCommand', bashCommand)
            repeat_err_msg = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
            error = b"dyld cache '(null)' not loaded: syscall to map cache into shared region failed"
            while (repeat_err_msg in error):
               process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
               output, error = process.communicate()
               n_repeat += 1

            if print_only:
               print(bashCommand)
            else:
               process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
               output, error = process.communicate()
            #if using the hst1pass_JA.e version, move the file to the correct location
            #move file to another place
            curr_file = JWST_image.replace('.fits',f'.{outname}')
            print('curr_file')
            if not python_jwst1pass:
               try:
                  os.rename(f'./{curr_file}',f'{output_dir}/{curr_file}')
               except:
                  pass
            print('curr_file path1', os.path.abspath(curr_file))
            # try:
            #    shutil.move(os.path.abspath(curr_file), output_dir)
            # except FileNotFoundError:
            #    print(f"Error: Source file '{source_file}' not found.")
            # print('curr_file path2', os.path.abspath(curr_file))
            # sdnvk
            
         if verbose:
            print('%s PERT PSF = %sx%s, Repeated %s times'%(JWST_image, pert_grid, pert_grid, n_repeat))

         remove_file(JWST_image.replace(jwst_im_type,'_psf'))
      else:

         if not os.path.isfile(psf_filename):
            print('WARNING: %s file not found!'%os.path.basename(psf_filename))
         if not os.path.isfile(gdc_filename):
            print('WARNING: %s file not found!'%os.path.basename(gdc_filename))
         print('Skipping %s.'%JWST_image)



def launch_jwst1pass_GH(flc_images, JWST_obs_to_use, JWST_path, exec_path, 
                        force_fmin = None, force_jwst1pass = True, verbose = True, 
                        n_processes = 1, auto_fmin_mult=1.0,jwst_im_type='_CAL',
                        python_jwst1pass=False):
   """
   This routine will launch jwst1pass_GH routine in parallel or serial 
   """
   from multiprocessing import Pool, cpu_count
   import os
   
   if python_jwst1pass:
      os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':' + os.path.abspath(os.path.join(exec_path, 'jwst1pass_py_v2'))

   print("\n---------------------------------")
   print("Finding sources in the JWST images")
   print("---------------------------------")

   args = []
   for JWST_image_obsid in [JWST_obs_to_use] if not isinstance(JWST_obs_to_use, list) else JWST_obs_to_use:
      for index_image, (obs_id, JWST_image) in flc_images.loc[flc_images['parent_obsid'] == JWST_image_obsid, ['obs_id', 'productFilename']].iterrows():
         entry = (JWST_path, exec_path, obs_id, JWST_image, force_fmin, force_jwst1pass, verbose, auto_fmin_mult, jwst_im_type, False, False, python_jwst1pass)
         # args.append(entry)
         if entry not in args:
            args.append(entry)
   
   if (len(args) > 1) and (n_processes != 1):
      pool = Pool(min(n_processes, len(args)))
      bashCommands = pool.map(jwst1pass_GH_multiproc_run, args)
      pool.close()
   
   else:
      bashCommands = []
      for arg in args:
         bashCommands.append(jwst1pass_GH_multiproc_run(arg))

   if verbose:
      print("\n-------------------------")
      print("List of jwst1pass commands")
      print("-------------------------\n")
      for command in bashCommands:
         print(command)
      print()

   # remove_file('LOG.psfperts.fits')
   # remove_file('fort.99')


def check_mat(mat_filename, iteration, min_stars_alignment = 100, alpha = 0.01, center_tolerance = 1e-3, plots = True, fix_mat = True, clipping_prob = 3, verbose= True):
   """
   This routine will read the transformation file MAT and provide a quality flag based on how Gaussian the transformation is.
   """
   if not os.path.isfile(mat_filename):
      return [False,False,False]
   mat = np.loadtxt(mat_filename)

   if fix_mat:
      clf = mixture.GaussianMixture(n_components = 1, covariance_type = 'spherical')
      clf.fit(mat[:, 6:8])
      log_prob = clf.score_samples(mat[:, 6:8])
      good_for_alignment = log_prob >= np.median(log_prob)-clipping_prob*np.std(log_prob)
      if verbose and (good_for_alignment.sum() < len(log_prob)): print('   Fixing MAT file for next iteration.')
      np.savetxt(mat_filename, mat[good_for_alignment, :], fmt='%12.4f')
   else:
      good_for_alignment = [True]*len(mat)

   with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      # Obtain the Shapiro–Wilk statistics
      stat, p = stats.shapiro(mat[:, 6:8])
   
   valid = [p >= alpha,  (mat[:, 6:8].mean(axis = 0) <= center_tolerance).all(), len(mat) >= min_stars_alignment]

   if plots:
      plt.close()
      fig, ax1 = plt.subplots(1, 1)
      try:
         ax1.plot(mat[~good_for_alignment,6], mat[~good_for_alignment,7], '.', ms = 2, label = 'Rejected')
      except:
         pass
      ax1.plot(mat[good_for_alignment,6], mat[good_for_alignment,7], '.', ms = 2, label = 'Used')
      ax1.axvline(x=0, linewidth = 0.75, color = 'k')
      ax1.axhline(y=0, linewidth = 0.75, color = 'k')
      ax1.set_xlabel(r'$X_{Gaia} - X_{HST}$ [pixels]')
      ax1.set_ylabel(r'$Y_{Gaia} - Y_{HST}$ [pixels]')
      ax1.grid()
      
      add_inner_title(ax1, 'Valid=%s\np=%.4f\ncen=(%.4f,%.4f)\nnum=%i'%(all(valid), p, mat[:, 6].mean(), mat[:, 7].mean(), len(mat)), 1)
      
      plt.savefig(mat_filename.split('.MAT')[0]+'_MAT_%i.png'%iteration, bbox_inches='tight')
      plt.close()

   return valid


def xym2pm_GH(iteration, Gaia_JWST_table_field, Gaia_JWST_table_filename, JWST_image_filename, lnk_filename, mat_filename, amp_filename, exec_path, date_reference_second_epoch, only_use_members, rewind_stars, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, previous_xym2pm, mat_plots, no_amplifier_based, min_stars_amp, use_stat):
   """
   This routine will execute xym2pm_GH Fortran routine using the correct arguments.
   """

   hdul = fits.open(JWST_image_filename)
   t_max = hdul[0].header['EXPEND']
   print('hdul t max', t_max)

   # Use WCS center from extension 1 if available, else fallback to TARG_RA/DEC
   if len(hdul) > 1 and 'CRVAL1' in hdul[1].header:
      ra_cent = hdul[1].header['CRVAL1']
      dec_cent = hdul[1].header['CRVAL2']
      xcen = hdul[1].header.get('CRPIX1', 1024.5)
      ycen = hdul[1].header.get('CRPIX2', 1024.5)
   else:
      ra_cent = hdul[0].header['TARG_RA']
      dec_cent = hdul[0].header['TARG_DEC']
      xcen = 5000.0
      ycen = 5000.0

   exptime = hdul[0].header['EFFINTTM']

   print(f'Working on {JWST_image_filename}. Exptime = {round(exptime,2)} sec')

   if 'FILTER' in hdul[0].header:
         filter = hdul[0].header['FILTER']
         if 'CLEAR' in filter:
            filter = hdul[0].header['PUPIL']
   else:
    filter = hdul[0].header['PUPIL']

   #date_reference_second_epoch is a string, default = J2016.0
   #Modified Julian Date (MJD) is a time format representing the number of days since midnight on November 17, 1858
   t_baseline = (date_reference_second_epoch.mjd - t_max) / 365.25
   print('t max', t_max)

   timeref = str(date_reference_second_epoch.jyear)
   match = pd.DataFrame()

   if force_pixel_scale is None:
      # Infer pixel scale from the header
    
      # poss_scales = []
      # try:
      #    curr_scale = proj_plane_pixel_scales(WCS(hdul[1].header)).mean()
      #    if curr_scale != 1.0:
      #       poss_scales.append(curr_scale)
      # except:
      #     pass
      poss_scales = [proj_plane_pixel_scales(WCS(hdul[1].header)).mean()]
        
      if len(poss_scales) > 0:
         pixel_scale = round(np.nanmedian(poss_scales)*3600,3)
      else:
         pixel_scale = round(1.0*3600,3)
    
      # print(HST_image_filename,poss_scales,pixel_scale*1000)
   else:
      pixel_scale = force_pixel_scale

   pixel_scale_mas = 1e3 * pixel_scale

   if ((iteration > 0) & (rewind_stars)):
      align_var = ['ra', 'ra_error', 'dec', 'dec_error', 'hst_gaia_pmra_%s'%use_stat, 'jwst_gaia_pmra_%s_error'%use_stat, 'jwst_gaia_pmdec_%s'%use_stat, 'jwst_gaia_pmdec_%s_error'%use_stat, 'gmag', 'use_for_alignment']
   else:
      align_var = ['ra', 'ra_error', 'dec', 'dec_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'gmag', 'use_for_alignment']

   if not os.path.isfile(lnk_filename) or not previous_xym2pm:

      f = open(Gaia_JWST_table_filename, 'w+')
      f.write('# ')
      Gaia_JWST_table_field.loc[:, align_var].astype({'use_for_alignment': 'int32'}).to_csv(f, index = False, sep = ' ', na_rep = 0)
      f.close()

      # Here it goes the executable line. Input values can be fine-tuned here
      # if (rewind_stars) & (iteration > 0):
      #    time = ' TIME=%s'%str(round(Time(t_max, format='mjd').jyear, 3))
      if (rewind_stars):
         time = ' TIME=%s'%str(round(Time(t_max, format='mjd').jyear, 3))
      else:
         time = ''

      if force_use_sat:
         use_sat = ' USESAT+'
      else:
         use_sat = ''

      if (fix_mat) and (iteration > 0):
         use_mat = " MAT=\"%s\" USEMAT+"%(mat_filename.split('./')[1])
      else:
         use_mat = ''

      if force_wcs_search_radius is None:
         use_brute = ''
      else:
         use_brute = ' BRUTE=%.1f'%force_wcs_search_radius

      if force_max_separation is None:
         max_separation = 5.0
      else:
         max_separation = force_max_separation

      if not no_amplifier_based:
         use_amp = ' NAMP=%i AMP+'%min_stars_amp
      else:
         use_amp = ''
      print('use_mat', " MAT=\"%s\" USEMAT+"%(mat_filename.split('./')[1]))

      # bashCommand = "%s/fortran_codes/xym2pm_GH.e %s %s RACEN=%f DECEN=%f XCEN=5000.0 YCEN=5000.0 PSCL=%s SIZE=10000 DISP=%.1f NMIN=%i TIMEREF=%s%s%s%s%s%s"%(exec_path, Gaia_JWST_table_filename, JWST_image_filename, ra_cent, dec_cent, pixel_scale, max_separation, min_stars_alignment, timeref, time, use_sat, use_mat, use_brute, use_amp)
      bashCommand = "%s/fortran_codes/xym2pm_GH.e %s %s RACEN=%f DECEN=%f XCEN=%f YCEN=%f PSCL=%s SIZE=10000 DISP=%.1f NMIN=%i TIMEREF=%s%s%s%s%s%s"%(exec_path, Gaia_JWST_table_filename, JWST_image_filename, ra_cent, dec_cent, xcen, ycen, pixel_scale, max_separation, min_stars_alignment, timeref, time, use_sat, use_mat, use_brute, use_amp)
      print('bashCommand', bashCommand)
      #bashCommand = "%s/fortran_codes/jwst1pass.e %s %s RACEN=%f DECEN=%f XCEN=5000.0 YCEN=5000.0 PSCL=%s SIZE=10000 DISP=%.1f NMIN=%i TIMEREF=%s%s%s%s%s%s"%(exec_path, Gaia_JWST_table_filename, JWST_image_filename, ra_cent, dec_cent, pixel_scale, max_separation, min_stars_alignment, timeref, time, use_sat, use_mat, use_brute, use_amp)

      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output, error = process.communicate()

      f = open(lnk_filename.replace(".LNK", "_6p_transformation.txt"), 'w+')
      f.write(output.decode("utf-8"))
      f.close()

#    try:
#       #Next are threshold rejection values for the MAT files. The alpha is the Shapiro–Wilk statistics. Lower values are more permissive.
#       if (iteration > 0) and only_use_members and not rewind_stars:
#          alpha = 1e-64
#          center_tolerance = 1e-2
#       else:
#          alpha = 0
#          center_tolerance = 1e-1

#       valid_mat = check_mat(mat_filename, iteration, min_stars_alignment, alpha = alpha, center_tolerance = center_tolerance, fix_mat = fix_mat, clipping_prob = 3., plots = mat_plots, verbose = verbose)

#       if all(valid_mat):

#          f = open(lnk_filename, 'r')
#          header = [w.replace('m_hst', filter) for w in f.readline().rstrip().strip("# ").split(' ')]
#          lnk = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_JWST_table_field.index)
#          f.close()
         
#          if os.path.isfile(amp_filename) and not no_amplifier_based:
#             f = open(amp_filename, 'r')
#             header = f.readline().rstrip().strip("# ").split(' ')
#             ampfile = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_JWST_table_field.index)
#             f.close()
#             lnk['xhst_gaia'] = ampfile.xhst_amp_gaia.values
#             lnk['yhst_gaia'] = ampfile.yhst_amp_gaia.values

#             f = open(lnk_filename.replace(".LNK", "_amp.LNK"), 'w+')
#             f.write('# ')
#             lnk.to_csv(f, index = False, sep = ' ', na_rep = 0)
#             f.close()

#          # Positional and mag error is know to be proportional to the QFIT parameter.
#          eradec_hst = lnk.q_hst.copy()
#          # Assign the maximum (worst) QFIT parameter to saturated stars.
#          eradec_hst[lnk.xhst_gaia.notnull()] = lnk[lnk.xhst_gaia.notnull()].q_hst.replace({np.nan:lnk.q_hst.mean()+5*lnk.q_hst.std()}).copy()

#          # 0.25 was chosen after empirical tests.
#          eradec_hst *= pixel_scale_mas * 0.25

#          lnk['relative_hst_gaia_pmra'] = -(lnk.x_gaia - lnk.xhst_gaia) * pixel_scale_mas / t_baseline
#          lnk['relative_hst_gaia_pmdec'] = (lnk.y_gaia - lnk.yhst_gaia) * pixel_scale_mas / t_baseline

#          # Notice the 1e3. xym2pm_GH.e takes the error in mas but returns it in arcsec.
#          lnk['relative_hst_gaia_pmra_error'] = eradec_hst / t_baseline
#          lnk['relative_hst_gaia_pmdec_error'] = eradec_hst / t_baseline
#          lnk['gaia_dra_uncertaintity'] = 1e3 * lnk.era_gaia / t_baseline
#          lnk['gaia_ddec_uncertaintity'] = 1e3 * lnk.edec_gaia / t_baseline
#          # Here we arbitrarily divided by 100 just for aesthetic reasons.
#          lnk['%s_error'%filter] = lnk.q_hst.replace({0:lnk.q_hst.max()}) * 0.01

#          match = lnk.loc[:, ['xc_hst', 'yc_hst',  filter, '%s_error'%filter, 'relative_hst_gaia_pmra', 'relative_hst_gaia_pmra_error', 'relative_hst_gaia_pmdec', 'relative_hst_gaia_pmdec_error', 'gaia_dra_uncertaintity', 'gaia_ddec_uncertaintity', 'q_hst']]

#          print('-->%s (%s, %ss): matched %i stars.'%(os.path.basename(JWST_image_filename), filter, exptime, len(match)))
      
#       else:
#          print('-->%s (%s, %ss): bad quality match:'%(os.path.basename(JWST_image_filename), filter, exptime))
#          if verbose:
#             if valid_mat[0] != True:
#                print('   Non Gaussian distribution found in the transformation')
#             if valid_mat[1] != True:
#                print('   Not (0,0) average of the distribution')
#             if valid_mat[2] != True:
#                print('   Less than %i stars used during the transformation'%min_stars_alignment)
#          print('   Skipping image.')
#          match = pd.DataFrame()
#    except:
#       if verbose:
#          print('-->%s  (%s, %ss): no match found.'%(os.path.basename(JWST_image_filename), filter, exptime))
#       match = pd.DataFrame()


    #Next are threshold rejection values for the MAT files. The alpha is the Shapiro–Wilk statistics. Lower values are more permissive.
      print("second lnk_filename", lnk_filename, os.path.isfile(lnk_filename), "mat_filename", os.path.isfile(mat_filename), "amp_filename", os.path.isfile(amp_filename))

      if (iteration > 0) and only_use_members and not rewind_stars:
         alpha = 1e-64
         center_tolerance = 1e-2
      else:
         alpha = 0
         center_tolerance = 1e-1

      valid_mat = check_mat(mat_filename, iteration, min_stars_alignment, alpha = alpha, center_tolerance = center_tolerance, fix_mat = fix_mat, clipping_prob = 3., plots = mat_plots, verbose = verbose)
      print('valid_mat', valid_mat)
      if all(valid_mat) and os.path.isfile(lnk_filename):

         f = open(lnk_filename, 'r')
         header = [w.replace('m_hst', filter) for w in f.readline().rstrip().strip("# ").split(' ')]
         lnk = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_JWST_table_field.index)
         f.close()

         if os.path.isfile(amp_filename) and not no_amplifier_based:
            f = open(amp_filename, 'r')
            header = f.readline().rstrip().strip("# ").split(' ')
            ampfile = pd.read_csv(f, names=header, sep = '\s+', comment='#', na_values = 0.0).set_index(Gaia_JWST_table_field.index)
            f.close()
            lnk['xhst_gaia'] = ampfile.xhst_amp_gaia.values
            lnk['yhst_gaia'] = ampfile.yhst_amp_gaia.values
            f = open(lnk_filename.replace(".LNK", "_amp.LNK"), 'w+')
            f.write('# ')
            lnk.to_csv(f, index = False, sep = ' ', na_rep = 0)
            f.close()
            
         # Positional and mag error is know to be proportional to the QFIT parameter.
         eradec_hst = lnk.q_hst.copy()
         # Assign the maximum (worst) QFIT parameter to saturated stars.
         eradec_hst[lnk.xhst_gaia.notnull()] = lnk[lnk.xhst_gaia.notnull()].q_hst.replace({np.nan:lnk.q_hst.mean()+5*lnk.q_hst.std()}).copy()

         # 0.25 was chosen after empirical tests.
         eradec_hst *= pixel_scale_mas * 0.25

         lnk['relative_hst_gaia_pmra'] = -(lnk.x_gaia - lnk.xhst_gaia) * pixel_scale_mas / t_baseline
         lnk['relative_hst_gaia_pmdec'] = (lnk.y_gaia - lnk.yhst_gaia) * pixel_scale_mas / t_baseline

         # Notice the 1e3. xym2pm_GH.e takes the error in mas but returns it in arcsec.
         lnk['relative_hst_gaia_pmra_error'] = eradec_hst / t_baseline
         lnk['relative_hst_gaia_pmdec_error'] = eradec_hst / t_baseline
         lnk['gaia_dra_uncertaintity'] = 1e3 * lnk.era_gaia / t_baseline
         lnk['gaia_ddec_uncertaintity'] = 1e3 * lnk.edec_gaia / t_baseline
         # Here we arbitrarily divided by 100 just for aesthetic reasons.
         #add replace NaN values 
         lnk['%s_error'%filter] = lnk.q_hst.replace({np.nan:lnk.q_hst.max()})
         lnk['%s_error'%filter] = lnk.q_hst.replace({0:lnk.q_hst.max()}) * 0.01
         print('nan?', lnk.q_hst.max())
         print("lnk['%s_error'%filter]", lnk['%s_error'%filter])

         match = lnk.loc[:, ['xc_hst', 'yc_hst',  filter, '%s_error'%filter, 'relative_hst_gaia_pmra', 'relative_hst_gaia_pmra_error', 'relative_hst_gaia_pmdec', 'relative_hst_gaia_pmdec_error', 'gaia_dra_uncertaintity', 'gaia_ddec_uncertaintity', 'q_hst']]

         print('-->%s (%s, %ss): matched %i stars.'%(os.path.basename(JWST_image_filename), filter, exptime, len(match)))

      else:
         print('-->%s (%s, %ss): bad quality match:'%(os.path.basename(JWST_image_filename), filter, exptime))
         if verbose:
            if valid_mat[0] != True:
               print('   Non Gaussian distribution found in the transformation')
            if valid_mat[1] != True:
               print('   Not (0,0) average of the distribution')
            if valid_mat[2] != True:
               print('   Less than %i stars used during the transformation'%min_stars_alignment)
         print('   Skipping image.')
         match = pd.DataFrame()
         
   return match


def xym2pm_GH_multiproc(args):
   """
   This routine pipes xym2pm_GH into multiple threads.
   """

   return xym2pm_GH(*args)


def launch_xym2pm_GH(Gaia_JWST_table, data_products_by_obs, JWST_obs_to_use, JWST_path, exec_path, date_reference_second_epoch, only_use_members = False, preselect_cmd = False, preselect_pm = False, rewind_stars = True, force_pixel_scale = None, force_max_separation = None, force_use_sat = True, fix_mat = True, no_amplifier_based = False, min_stars_amp = 25, force_wcs_search_radius = None, n_components = 1, clipping_prob = 6, use_only_good_gaia = False, min_stars_alignment = 100, use_stat = 'wmean', no_plots = False, verbose = True, quiet = False, ask_user_stop = False, max_iterations = 10, previous_xym2pm = False, remove_previous_files = True, n_processes = 1, plot_name = '', save_temporary_results = False, temporary_results_filename = ''):

   """
   This routine will launch xym2pm_GH Fortran routine in parallel or serial using the correct arguments.
   """
   from multiprocessing import Pool, cpu_count

   n_images = len(data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([JWST_obs_to_use] if not isinstance(JWST_obs_to_use, list) else JWST_obs_to_use), :])
   #isinstance() function returns True if the specified object is of the specified type
   #print("n images", n_images)
   if (n_images > 1) and (n_processes != 1):
      pool = Pool(min(n_processes, n_images))
      mat_plots = False
   else:
      mat_plots = ~no_plots

   if use_only_good_gaia & rewind_stars:
      Gaia_JWST_table['use_for_alignment'] = Gaia_JWST_table.clean_label.values and Gaia_JWST_table.pmra.notnull()
   elif rewind_stars:
      Gaia_JWST_table['use_for_alignment'] = Gaia_JWST_table.pmra.notnull()
   else:
      Gaia_JWST_table['use_for_alignment'] = True

   convergence = False
   iteration = 0

   pmra_evo = []
   pmdec_evo = []

   pmra_diff_evo = []
   pmdec_diff_evo = []

   hst_gaia_pmra_lsqt_evo = []
   hst_gaia_pmdec_lsqt_evo = []

   while not convergence:
      print("\n-----------")
      print("Iteration %i"%(iteration))
      print("-----------")
      
      # Close previous plots
      plt.close('all')

      args = []
      for index_image, (obs_id, JWST_image) in data_products_by_obs.loc[data_products_by_obs['parent_obsid'].isin([JWST_obs_to_use] if not isinstance(JWST_obs_to_use, list) else JWST_obs_to_use), ['obs_id', 'productFilename']].iterrows():
         #The iterrows() method generates an iterator object of the DataFrame, allowing us to iterate each row in the DataFrame
         JWST_image_filename = JWST_path+'mastDownload/JWST/'+obs_id+'/'+JWST_image
         Gaia_JWST_table_filename = JWST_path+'Gaia_%s.ascii'%JWST_image.split('.fits')[0]
         lnk_filename = JWST_image_filename.split('.fits')[0]+'.LNK'
         mat_filename = JWST_image_filename.split('.fits')[0]+'.MAT'
         amp_filename = JWST_image_filename.split('.fits')[0]+'.AMP'
         print("after define, lnk_filename", os.path.isfile(lnk_filename), "mat_filename", os.path.isfile(mat_filename), "amp_filename", os.path.isfile(amp_filename))
         if iteration == 0:
            if remove_previous_files:
               remove_file(mat_filename)
               remove_file(lnk_filename)
               remove_file(amp_filename)
               print('files removed')
            #print('Gaia_JWST_table (not Noun)', Gaia_JWST_table)
            #print("JWST_image_filename", JWST_image_filename, JWST_path)
            Gaia_JWST_table = find_stars_to_align(Gaia_JWST_table, JWST_image_filename)
            print('Gaia_JWST_table', Gaia_JWST_table)
            #print('sfgfv', Gaia_JWST_table['JWST_image'].str.contains(str(obs_id)), 'use_for_alignment')
            n_field_stars = Gaia_JWST_table.loc[Gaia_JWST_table['JWST_image'].str.contains(str(obs_id)), 'use_for_alignment'].count()
            print('n_field_stars', n_field_stars)

         u_field_stars = Gaia_JWST_table.loc[Gaia_JWST_table['JWST_image'].str.contains(str(obs_id)), 'use_for_alignment'].sum()
         print('u_field_stars', u_field_stars)
         # We assume that the some parts of the image can have slightly lower stars density than others.
         # Therefore, we require 5 times the number of stars per amplifier in the entire image instead of 4.
         if (u_field_stars < min_stars_amp*5) and (no_amplifier_based == False):
            print('WARNING: Not enough stars for alignment in %s as to separate amplifiers. Only one channel will be used.'%JWST_image)
            no_amplifier_based_inuse = True
         elif (u_field_stars >= min_stars_amp*5) and (no_amplifier_based == False):
            no_amplifier_based_inuse = False
         else:
            no_amplifier_based_inuse = True

         if (u_field_stars < min_stars_alignment):
            print('WARNING: Not enough member stars in %s. Using all the stars in the field.'%JWST_image)
            Gaia_JWST_table.loc[Gaia_JWST_table['JWST_image'].str.contains(str(obs_id)), 'use_for_alignment'] = True

         Gaia_JWST_table_field = Gaia_JWST_table.loc[Gaia_JWST_table['JWST_image'].str.contains(str(obs_id)), :]

         args.append((iteration, Gaia_JWST_table_field, Gaia_JWST_table_filename, JWST_image_filename, lnk_filename, mat_filename, amp_filename, exec_path, date_reference_second_epoch, only_use_members, rewind_stars, force_pixel_scale, force_max_separation, force_use_sat, fix_mat, force_wcs_search_radius, min_stars_alignment, verbose, previous_xym2pm, mat_plots, no_amplifier_based_inuse, min_stars_amp, use_stat))

      if (len(args) > 1) and (n_processes != 1):
         lnks = pool.map(xym2pm_GH_multiproc, args)
        
      else:
         lnks = []
         for arg in args:
            lnks.append(xym2pm_GH_multiproc(arg))

      lnks = pd.concat(lnks, sort=True)

      if len(lnks) == 0:
         print('WARNING: No match could be found for any of the images. Please try with other parameters.\nExiting now.\n')
         remove_file(exec_path)
         sys.exit(1)
      else:
         print("-----------")
         print('%i stars were used in the transformation.'%(min([Gaia_JWST_table.use_for_alignment.sum(), n_field_stars])))

      lnks_averaged = lnks.groupby(lnks.index).apply(weighted_avg_err)

      # Gaia positional errors have to be added in quadrature
      for use_stat_i in ['mean', 'wmean', 'median']:
         lnks_averaged['relative_hst_gaia_pmra_%s_error'%use_stat_i] = np.sqrt(lnks_averaged['relative_hst_gaia_pmra_%s_error'%use_stat_i]**2 + lnks_averaged['gaia_dra_uncertaintity_mean']**2)
         lnks_averaged['relative_hst_gaia_pmdec_%s_error'%use_stat_i] = np.sqrt(lnks_averaged['relative_hst_gaia_pmdec_%s_error'%use_stat_i]**2 + lnks_averaged['gaia_ddec_uncertaintity_mean']**2)

      # Remove redundant columns
      lnks_averaged = lnks_averaged.drop(columns=[col for col in lnks_averaged if col.startswith('gaia') or (col.startswith('q_jwst') and 'wmean' in col)])

      try:
         Gaia_JWST_table.drop(columns = lnks_averaged.columns, inplace = True)
      except:
         pass

      # Obtain absolute PMs
      Gaia_JWST_table = absolute_pm(Gaia_JWST_table.join(lnks_averaged), use_only_good_gaia = use_only_good_gaia, clipping_prob = clipping_prob, verbose = verbose,  no_plots = no_plots, plot_name = plot_name, iteration = iteration)

      # Ir rewind is in use, we complete the PMs from Gaia with those measured in the first iteration.
      if (rewind_stars) & (iteration == 0):
         Gaia_JWST_table.loc[Gaia_JWST_table.pmra.notnull(), 'hst_gaia_pmra_%s'%use_stat] = Gaia_JWST_table.loc[Gaia_JWST_table.pmra.notnull(), 'pmra'].values
         Gaia_JWST_table.loc[Gaia_JWST_table.pmdec.notnull(), 'hst_gaia_pmdec_%s'%use_stat] = Gaia_JWST_table.loc[Gaia_JWST_table.pmdec.notnull(), 'pmdec'].values

      # Membership selection
      if only_use_members:
         if (preselect_cmd == True) and (no_plots == False) and (quiet == False):
            if (iteration == 0):
                  Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'clustering_cmd'] =  manual_select_from_cmd(Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'bp_rp'].rename('BP-RP'), Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'gmag'].rename('Gmag'))

         else:
            Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'clustering_cmd'] = True

         if (preselect_pm == True) and (no_plots == False) and (quiet == False):
            if (iteration == 0):
               Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'clustering_pm'] =  manual_select_from_pm(Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'relative_hst_gaia_pmra_%s'%use_stat], Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'relative_hst_gaia_pmdec_%s'%use_stat])
               gauss_center = Gaia_JWST_table.loc[Gaia_JWST_table.clustering_pm == True, ['relative_hst_gaia_pmra_%s'%use_stat, 'relative_hst_gaia_pmdec_%s'%use_stat]].mean().values

         else:
            Gaia_JWST_table.loc[Gaia_JWST_table.use_for_alignment, 'clustering_pm'] = True
            gauss_center = [0,0]

         clustering_data = Gaia_JWST_table.clustering_cmd & Gaia_JWST_table.clustering_pm

         # Select stars in the PM space asuming spherical covariance (Reasonable for dSphs and globular clusters) 
         pm_clustering = pm_cleaning_GMM_recursive(Gaia_JWST_table.copy(), ['relative_hst_gaia_pmra_%s'%use_stat, 'relative_hst_gaia_pmdec_%s'%use_stat], ['relative_hst_gaia_pmra_%s_error'%use_stat, 'relative_hst_gaia_pmdec_%s_error'%use_stat], clustering_data = clustering_data, data_0 = gauss_center, n_components = n_components, covariance_type = 'full', clipping_prob = clipping_prob, verbose = verbose,  no_plots = no_plots, plot_name = '%s_%i.png'%(plot_name, iteration))
      else:
         clustering_data = pd.Series(data= [True]*len(Gaia_JWST_table))
         pm_clustering = pd.Series(data= [True]*len(Gaia_JWST_table))

      if (not only_use_members) and (not rewind_stars):
         convergence = True

      # GaiaHub will only use stars with Gaia PMs to set up the reference frames, and those that pased the selection
      Gaia_JWST_table['use_for_alignment'] = pm_clustering & clustering_data & Gaia_JWST_table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat]

      # Useful statistics:
      id_pms = np.isfinite(Gaia_JWST_table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(Gaia_JWST_table['hst_gaia_pmdec_%s'%use_stat])

      pmra_evo.append(Gaia_JWST_table.loc[id_pms, 'hst_gaia_pmra_%s'%use_stat])
      pmdec_evo.append(Gaia_JWST_table.loc[id_pms, 'hst_gaia_pmdec_%s'%use_stat])

      hst_gaia_pmra_lsqt_evo.append(np.nanstd( (Gaia_JWST_table.loc[id_pms, 'hst_gaia_pmra_%s'%use_stat]  - Gaia_JWST_table.loc[id_pms, 'pmra'])))
      hst_gaia_pmdec_lsqt_evo.append(np.nanstd( (Gaia_JWST_table.loc[id_pms, 'hst_gaia_pmdec_%s'%use_stat] - Gaia_JWST_table.loc[id_pms, 'pmdec'])))
      print('RMS(PM_HST+Gaia - PM_Gaia) = (%.4e, %.4e) m.a.s.' %(hst_gaia_pmra_lsqt_evo[-1], hst_gaia_pmdec_lsqt_evo[-1]))

      if iteration >= (max_iterations-1):
         print('\nWARNING: Max number of iterations reached: Something might have gone wrong!\nPlease check the results carefully.')
         convergence = True

      elif iteration > 0:
         pmra_diff_evo.append(np.nanmean(pmra_evo[-1] - pmra_evo[-2]))
         pmdec_diff_evo.append(np.nanmean(pmdec_evo[-1] - pmdec_evo[-2]))

         # If rewind_stars is True, the code will converge when the difference between interations is smaller than the error in PMs
         threshold = np.nanmean(Gaia_JWST_table.loc[id_pms, ['relative_hst_gaia_pmra_%s_error'%use_stat, 'relative_hst_gaia_pmdec_%s_error'%use_stat]].mean())*1e-2

         print('PM variation = (%.4e, %.4e)  m.a.s.' %(pmra_diff_evo[-1], pmdec_diff_evo[-1]))

         if rewind_stars:
             if iteration > 1:
                print('Convergence threshold = %.4e  m.a.s.'%threshold)
                convergence = (np.abs(pmra_diff_evo[-1]) <= threshold) & (np.abs(pmdec_diff_evo[-1]) <= threshold)
             else:
                convergence = False
         else:
            convergence = Gaia_JWST_table.use_for_alignment.equals(previous_use_for_alignment)

      if ask_user_stop & (iteration > 0) & (quiet == False):
         with plt.rc_context(rc={'interactive': False}):
            plt.gcf().show()
         try:
            print('\nCheck the preliminary results in the VPD.')
            continue_loop = input('Continue with the next iteration? ') or 'y'
            continue_loop = str2bool(continue_loop)
            print('')
            if not continue_loop:
               convergence = True
         except:
            print('WARNING: Answer not understood. Continuing execution.')

      previous_use_for_alignment = Gaia_JWST_table.use_for_alignment.copy()

      if save_temporary_results:
          Gaia_JWST_table[Gaia_JWST_table['relative_hst_gaia_pmdec_%s'%use_stat].notnull() & Gaia_JWST_table['relative_hst_gaia_pmra_%s'%use_stat].notnull()].to_csv(temporary_results_filename.split('.csv')[0]+'_%i.csv'%iteration)

      iteration += 1

   if (n_images > 1) and (n_processes != 1):
      pool.close()

   Gaia_JWST_table = Gaia_JWST_table[Gaia_JWST_table['relative_hst_gaia_pmdec_%s'%use_stat].notnull() & Gaia_JWST_table['relative_hst_gaia_pmra_%s'%use_stat].notnull()]

   pmra_evo = np.array(pmra_evo).T
   pmdec_evo = np.array(pmdec_evo).T

   hst_gaia_pm_lsqt_evo = np.array([hst_gaia_pmra_lsqt_evo, hst_gaia_pmdec_lsqt_evo]).T
   pm_diff_evo = np.array([pmra_diff_evo, pmdec_diff_evo]).T

   if (iteration > 1) and (no_plots == False):
      try:
         plt.close()
         fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True)

         ax1.plot(np.arange(iteration-1)+1, np.abs(pm_diff_evo[:,0]), '-', label = r'$\Delta(\mu_{\alpha *})$')
         ax1.plot(np.arange(iteration-1)+1, np.abs(pm_diff_evo[:,1]), '-', label = r'$\Delta(\mu_{\delta})$')

         ax2.plot(np.arange(iteration), hst_gaia_pm_lsqt_evo[:,0], '-', label = r'RMS$(\mu_{\alpha *, HST+Gaia} - \mu_{\alpha *, Gaia})$')
         ax2.plot(np.arange(iteration), hst_gaia_pm_lsqt_evo[:,1], '-', label = r'RMS$(\mu_{\delta, HST+Gaia} - \mu_{\delta, Gaia})$')

         ax1.axhline(y=threshold, linewidth = 0.75, color = 'k')
         ax1.set_ylabel(r'$\Delta(\mu)$')
         ax2.set_ylabel(r'RMS$(\mu_{HST-Gaia} - \mu_{Gaia})$')
         ax2.set_xlabel(r'iteration #')

         ax1.grid()
         ax2.grid()

         ax1.legend(shadow=True, fancybox=True)
         ax2.legend(shadow=True, fancybox=True)

         plt.savefig('%s_PM_RMS_iterations_%s.pdf'%(plot_name, use_stat), bbox_inches='tight')
         plt.close()
      except:
         pass

   return Gaia_JWST_table, lnks


def find_stars_to_align(stars_catalog, JWST_image_filename):
   """
   This routine will find which stars from stars_catalog within and HST image.
   """

   from shapely.geometry.polygon import Polygon as shap_polygon
   from shapely.geometry import Point
   from shapely.ops import unary_union

   JWST_image = JWST_image_filename.split('/')[-1].split('.fits')[0]
   print("JWST_image_filename", JWST_image_filename)
   # sufvnhbi
   if not os.path.isfile(JWST_image_filename):
      #Check if a path refers to a regular file (not a directory or any other type of file system object)
      print(f'RRR SKIPPING image {JWST_image} because no JWST image was found')
      return
   hdu = fits.open(JWST_image_filename)
   try:
      len(hdu)
   except:
      print(f'SKIPPING image {JWST_image} because no good JWST image was found')
      return
   
   if 'JWST_image' not in stars_catalog.columns:
      stars_catalog['JWST_image'] = ""
   print("star catalog, JWST_image", stars_catalog['JWST_image'], "hdu", hdu)

   idx_Gaia_in_field = []
   footprint = []
   #if ii > len(hdu)-1:
      #print(f'1Skipping header index {ii} for image {JWST_image_filename}')
      #continue

   try:
      curr_scale = proj_plane_pixel_scales(WCS(hdu[1].header)).mean()
      if curr_scale == 1.0:
        #bad index
        print(f'2Skipping header index {1} for image {JWST_image_filename}')
        #continue
   except:
      pass

   #print('hdu[ii].header', hdu[1].header)
   #print('WCS(hdu[ii].header)', WCS(hdu[1].header))
   #print('proj_plane_..', proj_plane_pixel_scales(WCS(hdu[1].header)))
        
   try:
      wcs = WCS(hdu[1].header)
      footprint_chip = wcs.calc_footprint()
   except:
      print(ii,JWST_image_filename)
      wcs = WCS(hdu[1].header)
      footprint_chip = wcs.calc_footprint()
   #print('footprint_chip', footprint_chip)
      # We add 10 arcsec of HST pointing error to the footprint to ensure we have all the stars.
   center_chip = np.mean(footprint_chip, axis = 0)

   footprint_chip[np.where(footprint_chip[:,0] < center_chip[0]),0] -= 0.0028 / np.cos(np.deg2rad(center_chip[1]))
   footprint_chip[np.where(footprint_chip[:,0] > center_chip[0]),0] += 0.0028 / np.cos(np.deg2rad(center_chip[1]))

   footprint_chip[np.where(footprint_chip[:,1] < center_chip[1]),1] -= 0.0028
   footprint_chip[np.where(footprint_chip[:,1] > center_chip[1]),1] += 0.0028

   tuples_coo = [(ra % 360, dec) for ra, dec in zip(footprint_chip[:, 0], footprint_chip[:, 1])]
   footprint.append(shap_polygon(tuples_coo))
   
   footprint = unary_union(footprint)

   for idx, ra, dec in zip(stars_catalog.index, stars_catalog.ra, stars_catalog.dec):
      if Point(ra, dec).within(footprint):
         idx_Gaia_in_field.append(idx)

   stars_catalog.loc[idx_Gaia_in_field, 'JWST_image'] = stars_catalog.loc[idx_Gaia_in_field, 'JWST_image'].astype(str) + '%s '%JWST_image

   return stars_catalog


def weighted_avg_err(table):
   """
   Weighted average its error and the standard deviation.
   """

   var_cols = [x for x in table.columns if not '_error' in x]
   var_cols_err = ['%s_error'%col for col in var_cols]

   x_i = table.loc[:, var_cols]
   #print('x_i', len(x_i), var_cols)
   ex_i = table.reindex(columns = var_cols_err)
   ex_i.columns = ex_i.columns.str.rstrip('_error')

   weighted_variance = (1./(1./ex_i**2).sum(axis = 0))
   weighted_avg = ((x_i.div(ex_i**2)).sum(axis = 0) * weighted_variance).add_suffix('_wmean')
   weighted_avg_error = np.sqrt(weighted_variance[~weighted_variance.index.duplicated()]).add_suffix('_wmean_error')

   avg = x_i.mean().add_suffix('_mean')
   #print('np.sqrt(len(x_i)', np.sqrt(len(x_i)))
   avg_error = x_i.std().add_suffix('_mean_error')/np.sqrt(len(x_i))
   std = x_i.std().add_suffix('_std')

   # The formula is 1.25*sigma/sqrt(N) for medians of normal distributions. It gets better from there. This is the upper limmit of error.
   median_err = (1.25404*x_i.std()/np.sqrt(len(x_i))).add_suffix('_median_error')
   median = x_i.median().add_suffix('_median')

   return pd.concat([weighted_avg, weighted_avg_error, avg, avg_error, median, median_err, std])


def absolute_pm(table, use_only_good_gaia = True, n_components = 1, covariance_type = 'full', clipping_prob = 3, no_plots = True, verbose = True, plot_name = '', iteration = ''):
   """
   This routine computes the absolute PM just adding the absolute differences between Gaia and HST PMs.
   """   

   # We try to select only good Gaia stars to derive the absolute reference frame:

   for use_stat in ['mean', 'wmean', 'median']:

      # We try to select only good Gaia-HST stars to derive the absolute reference frame:
      hst_gaia_error_threshold = table.loc[:, ['relative_hst_gaia_pmra_%s_error'%use_stat, 'relative_hst_gaia_pmdec_%s_error'%use_stat]].quantile(q = 0.85)
      Good_hst_gaia = (table.loc[:, ['relative_hst_gaia_pmra_%s_error'%use_stat, 'relative_hst_gaia_pmdec_%s_error'%use_stat]] <= hst_gaia_error_threshold).all(axis = 1)

      if use_only_good_gaia:
         Good_stars = Good_gaia & Good_hst_gaia
      else:
         Good_stars = Good_hst_gaia

      pm_differences = table.loc[Good_stars, ['pmra', 'pmdec']].copy() - table.loc[Good_stars, ['relative_hst_gaia_pmra_%s'%use_stat, 'relative_hst_gaia_pmdec_%s'%use_stat]].values
      pm_differences_error = np.sqrt(table.loc[Good_stars, ['pmra_error', 'pmdec_error']].copy()**2 + table.loc[Good_stars, ['relative_hst_gaia_pmra_%s_error'%use_stat, 'relative_hst_gaia_pmdec_%s_error'%use_stat]].values**2)

      ref = pm_differences.join(pm_differences_error)

      try:
         pm_clustering = pm_cleaning_GMM_recursive(ref, ['pmra', 'pmdec'], ['pmra_error', 'pmdec_error'], data_0 = pm_differences.mean(axis = 0), n_components = n_components, covariance_type = 'full', clipping_prob = clipping_prob, verbose = verbose,  no_plots = no_plots, plot_name = '%s_%i_absolute_%s.png'%(plot_name, iteration, use_stat))
      except:
         print('Something went wrong... using all the stars to measure the %s absolute reference frame.'%use_stat)
         pm_clustering = [True]*len(ref)

      pm_differences = weighted_avg_err(ref.loc[pm_clustering, :])

      table['hst_gaia_pmra_%s'%use_stat], table['hst_gaia_pmdec_%s'%use_stat] = table['relative_hst_gaia_pmra_%s'%use_stat] + pm_differences['pmra_%s'%use_stat], table['relative_hst_gaia_pmdec_%s'%use_stat] + pm_differences['pmdec_%s'%use_stat]
      table['hst_gaia_pmra_%s_error'%use_stat], table['hst_gaia_pmdec_%s_error'%use_stat] = np.sqrt(table['relative_hst_gaia_pmra_%s_error'%use_stat]**2 + pm_differences['pmra_%s_error'%use_stat]**2), np.sqrt(table['relative_hst_gaia_pmdec_%s_error'%use_stat]**2 + pm_differences['pmdec_%s_error'%use_stat]**2)

      table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] = False
      table.loc[Good_stars, 'use_for_absolute_ref_frame_%s'%use_stat] = pm_clustering

   return table


def cli_progress_test(current, end_val, bar_length=50):
   """
   Just a progress bar.
   """

   percent = float(current) / end_val
   hashes = '#' * int(round(percent * bar_length))
   spaces = ' ' * (bar_length - len(hashes))
   sys.stdout.write("\rProcessing: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
   sys.stdout.flush()


def add_inner_title(ax, title, loc, size=None, **kwargs):
   """
   Add text with stroke inside plots 
   """
   from matplotlib.offsetbox import AnchoredText
   from matplotlib.patheffects import withStroke

   if size is None:
       size = dict(size=plt.rcParams['legend.fontsize'])
   at = AnchoredText(title, loc=loc, prop=size,
                     pad=0., borderpad=0.5,
                     frameon=False, **kwargs)
   ax.add_artist(at)
   at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
   return at


def bin_errors(x, y, y_error, n_bins = 10):
   """
   Binned statistics
   """

   bins = np.quantile(x, np.linspace(0,1,n_bins+1))
   if bins[1] == bins[0]:
      bins = bins[0]+np.linspace(0,0.001,n_bins+1)
   print('bins', bins)
   bins_centers = (bins[:-1] + bins[1:]) / 2

   mean = np.full(n_bins, np.nan)
   std = np.full(n_bins, np.nan)
   mean_error = np.full(n_bins, np.nan)
 
   try:
       mean = stats.binned_statistic(x, y, statistic='mean', bins=bins).statistic
   except:
       print('from except x notice for mean', x, 'y', y)
       
   try:
       std = stats.binned_statistic(x, y, statistic='std', bins=bins).statistic
   except:
       print('from except x notice for std', x, 'y', y)
   try:
       mean_error = stats.binned_statistic(x, y_error, statistic='mean', bins=bins).statistic
   except:
       print('from except x notice for mean_error', x, 'y', y)

   return bins_centers, mean, std, mean_error
   

def plot_results(table, lnks, jwst_image_list, HST_path, avg_pm, use_stat = 'wmean', use_members = True, plot_name_1 = 'output1', plot_name_2 = 'output2', plot_name_3 = 'output3', plot_name_4 = 'output4', plot_name_5 = 'output5', plot_name_6 = 'output6', plot_name_7 = 'output7', ext = '.pdf'):
   """
   Plot results
   """
   
   GDR = 'Gaia'
   GaiaJWST_GDR = 'GaiaJWST'
   sigma_lims = 3

   pmra_lims = [table['hst_gaia_pmra_%s'%use_stat].mean()-sigma_lims*table['hst_gaia_pmra_%s'%use_stat].std(), table['hst_gaia_pmra_%s'%use_stat].mean()+sigma_lims*table['hst_gaia_pmra_%s'%use_stat].std()]
   pmdec_lims = [table['hst_gaia_pmdec_%s'%use_stat].mean()-sigma_lims*table['hst_gaia_pmdec_%s'%use_stat].std(), table['hst_gaia_pmdec_%s'%use_stat].mean()+sigma_lims*table['hst_gaia_pmdec_%s'%use_stat].std()]
   
   # Plot the VPD and errors
   
   plt.close('all')
   
   fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = False, sharey = False, figsize = (10, 3))

   ax1.plot(table.pmra[table.use_for_alignment == False], table.pmdec[table.use_for_alignment == False], 'k.', ms = 1, alpha = 0.35)
   ax1.plot(table.pmra[table.use_for_alignment == True], table.pmdec[table.use_for_alignment == True], 'k.', ms = 1)
   ax1.grid()
   ax1.set_xlabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmdec_lims)
   except:
      pass
   add_inner_title(ax1, GDR, loc=1)


   ax2.plot(table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False], 'r.', ms =1, alpha = 0.35)
   ax2.plot(table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True], 'r.', ms =1)
   ax2.grid()
   ax2.set_xlabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmra_lims)
      ax2.set_ylim(pmdec_lims)
   except:
      pass
   add_inner_title(ax2, GaiaJWST_GDR, loc=1)

   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(0.5*table.pmra_error[table.use_for_alignment == False]**2 + 0.5*table.pmdec_error[table.use_for_alignment == False]**2), 'k.', ms = 1, alpha = 0.35)
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(0.5*table.pmra_error[table.use_for_alignment == True]**2 + 0.5*table.pmdec_error[table.use_for_alignment == True]**2), 'k.', ms = 1, label = GDR)
   print('check values', table.pmdec_error[table.use_for_alignment == True])
   ax3.plot(table.gmag[table.use_for_alignment == False], np.sqrt(0.5*table['hst_gaia_pmra_%s_error'%use_stat][table.use_for_alignment == False]**2 + 0.5*table['hst_gaia_pmdec_%s_error'%use_stat][table.use_for_alignment == False]**2), 'r.', ms = 1, alpha = 0.35)
    
   ax3.plot(table.gmag[table.use_for_alignment == True], np.sqrt(0.5*table['hst_gaia_pmra_%s_error'%use_stat][table.use_for_alignment == True]**2 + 0.5*table['hst_gaia_pmdec_%s_error'%use_stat][table.use_for_alignment == True]**2), 'r.', ms = 1, label = GaiaJWST_GDR)

   ax3.grid()
   ax3.legend(prop={'size': 8})

   try:
      ax3.set_ylim(0, np.sqrt(table.pmra_error**2+table.pmdec_error**2).max())
   except:
      pass
   ax3.set_xlabel(r'$G$')
   ax3.set_ylabel(r'$\Delta\mu$ [m.a.s./yr.]')

   plt.subplots_adjust(wspace=0.3, hspace=0.1)

   plt.savefig(plot_name_1+ext, bbox_inches='tight')

   plt.close('all')


   # Plot the difference between Gaia and HST+Gaia

   fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex = False, sharey = False, figsize = (10, 3))
   ax1.errorbar(table.pmra, table['hst_gaia_pmra_%s'%use_stat], xerr=table.pmra_error, yerr=table['hst_gaia_pmra_%s_error'%use_stat], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)

   ax1.scatter(table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'pmra'], table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'hst_gaia_pmra_%s'%use_stat], s = 4, linewidth = 1, facecolor='g', edgecolor='g', zorder = 2)

   ax1.plot([pmra_lims[0], pmra_lims[1]], [pmra_lims[0], pmra_lims[1]], 'r-', linewidth = 0.5)
   ax1.grid()
   ax1.set_xlabel(r'$\mu_{\alpha*, Gaia}$ [m.a.s./yr.]')
   ax1.set_ylabel(r'$\mu_{\alpha*, HST + Gaia }$ [m.a.s./yr.]')
   try:
      ax1.set_xlim(pmra_lims)
      ax1.set_ylim(pmra_lims)      
   except:
      pass

   ax2.errorbar(table.pmdec, table['hst_gaia_pmdec_%s'%use_stat], xerr=table.pmdec_error, yerr=table['hst_gaia_pmdec_%s_error'%use_stat], fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)

   ax2.scatter(table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'pmdec'], table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'hst_gaia_pmdec_%s'%use_stat], s = 4, linewidth = 1, facecolor='g', edgecolor='g', zorder = 2)

   ax2.plot([pmdec_lims[0], pmdec_lims[1]], [pmdec_lims[0], pmdec_lims[1]], 'r-', linewidth = 0.5)
   ax2.grid()
   ax2.set_xlabel(r'$\mu_{\delta, Gaia}$ [m.a.s./yr.]')
   ax2.set_ylabel(r'$\mu_{\delta, HST + Gaia}$ [m.a.s./yr.]')
   try:
      ax2.set_xlim(pmdec_lims)
      ax2.set_ylim(pmdec_lims)      
   except:
      pass

   ax3.errorbar(table.pmra - table['hst_gaia_pmra_%s'%use_stat], table.pmdec - table['hst_gaia_pmdec_%s'%use_stat], xerr=np.sqrt(table.pmra_error**2 + table['hst_gaia_pmra_%s_error'%use_stat]**2), yerr=np.sqrt(table.pmdec_error**2 + table['hst_gaia_pmdec_%s_error'%use_stat]**2), fmt = '.', ms=2, color = '0.1', zorder = 1, alpha = 0.5, elinewidth = 0.5)

   ax3.scatter(table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'pmra'] - table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'hst_gaia_pmra_%s'%use_stat], table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat] == True, 'pmdec'] - table.loc[table.loc[:, 'use_for_absolute_ref_frame_%s'%use_stat]==True, 'hst_gaia_pmdec_%s'%use_stat], s = 4, facecolor='g', edgecolor='g', zorder = 2, label = 'abs. ref. frame')

   ax3.axvline(x=0, color ='r', linewidth = 0.5)
   ax3.axhline(y=0, color ='r', linewidth = 0.5)

   ax3.grid()
   ax3.set_aspect('equal', adjustable='datalim')
   
   ax3.set_xlabel(r'$\mu_{\alpha*, Gaia}$ - $\mu_{\alpha*, HST + Gaia}$ [m.a.s./yr.]')
   ax3.set_ylabel(r'$\mu_{\delta, Gaia}$ - $\mu_{\delta, HST + Gaia}$ [m.a.s./yr.]')
   ax3.legend(prop={'size': 8})

   plt.subplots_adjust(wspace=0.3, hspace=0.1)

   plt.savefig(plot_name_2+ext, bbox_inches='tight')


   plt.close('all')

   # Plot the CMD

   fig, ax = plt.subplots(1,1, sharex = False, sharey = False, figsize = (5., 5.))

   ax.plot(table.loc[table.use_for_alignment == False, 'bp_rp'], table.loc[table.use_for_alignment == False, 'gmag'], 'k.', ms=2, alpha = 0.35)
   ax.plot(table.loc[table.use_for_alignment == True, 'bp_rp'], table.loc[table.use_for_alignment == True, 'gmag'], 'k.', ms=3)

   ax.set_ylabel('G')
   ax.set_xlabel('BP-RP')

   try:
      ax.set_ylim(np.nanmax(table.loc[:, 'gmag'])+0.5, np.nanmin(table.loc[:, 'gmag'])-0.5)
   except:
      pass

   ax.grid()
   
   plt.savefig(plot_name_3+ext, bbox_inches='tight')

   plt.close('all')


   # Plot the sky projection

   hdu_list = []
   for index_image, (obs_id, JWST_image) in jwst_image_list.loc[:, ['obs_id', 'productFilename']].iterrows():

      JWST_image_filename = HST_path+'mastDownload/JWST/'+obs_id+'/'+JWST_image

      hdu_list.append(fits.open(JWST_image_filename)[1])

   if len(hdu_list) > 1:
      from reproject.mosaicking import find_optimal_celestial_wcs
      from reproject import reproject_interp
      from reproject.mosaicking import reproject_and_coadd
      
      wcs, shape = find_optimal_celestial_wcs(hdu_list, resolution=0.5 * u.arcsec)
      image_data, image_footprint = reproject_and_coadd(hdu_list, wcs, shape_out=shape, reproject_function=reproject_interp, order = 'nearest-neighbor', match_background=True)

   else:
      wcs = WCS(hdu_list[0].header)
      image_data = hdu_list[0].data

   # Identify stars with Gaia PMs
   id_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec'])
   id_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_stat])

   fig = plt.figure(figsize=(5., 5.), dpi = 250)
   ax = fig.add_subplot(111, projection=wcs)

   norm = ImageNormalize(image_data, interval = ManualInterval(0.0,0.15))

   im = ax.imshow(image_data, cmap='gray_r', origin='lower', norm=norm, zorder = 0)

   ax.set_xlabel("RA")
   ax.set_ylabel("Dec")

   if (table.use_for_alignment.sum() < len(table)) & (table.use_for_alignment.sum() > 0):
      try:
         p1 = ax.scatter(table['ra'][table.use_for_alignment == False & id_gaia], table['dec'][table.use_for_alignment == False & id_gaia], transform=ax.get_transform('world'), s=10, linewidth = 1, facecolor='none', edgecolor='k', alpha = 0.35, zorder = 1)
         p1 = ax.scatter(table['ra'][table.use_for_alignment == False & id_hst_gaia], table['dec'][table.use_for_alignment == False & id_hst_gaia], transform=ax.get_transform('world'), s=30, linewidth = 1, facecolor='none', edgecolor='r', alpha = 0.35, zorder = 2)
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q1 = ax.quiver(table['ra'][table.use_for_alignment == False & id_hst_gaia], table['dec'][table.use_for_alignment == False & id_hst_gaia], -table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == False & id_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False & id_hst_gaia], transform=ax.get_transform('world'), width = 0.003, angles = 'xy', color='r', alpha = 0.35, zorder = 2)
      except:
         pass
   if table.use_for_alignment.sum() > 0:
      try:
         p2 = ax.scatter(table['ra'][table.use_for_alignment == True & id_gaia], table['dec'][table.use_for_alignment == True & id_gaia], transform=ax.get_transform('world'), s=10, linewidth = 1, facecolor='none', label=GDR, edgecolor='k', zorder = 1)
         p2 = ax.scatter(table['ra'][table.use_for_alignment == True & id_hst_gaia], table['dec'][table.use_for_alignment == True & id_hst_gaia], transform=ax.get_transform('world'), s=30, linewidth = 1, facecolor='none', label=GaiaJWST_GDR, edgecolor='r', zorder = 2)
      except:
         pass
      try:
         with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q2 = ax.quiver(table['ra'][table.use_for_alignment == True & id_hst_gaia],table['dec'][table.use_for_alignment == True & id_hst_gaia], -table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True & id_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True & id_hst_gaia], transform=ax.get_transform('world'), width = 0.003, angles = 'xy', color='r', zorder = 2)
      except:
         pass

   ax.grid()

   plt.legend()
   
   plt.tight_layout()
   plt.savefig(plot_name_4+'.pdf', bbox_inches='tight')
   
   plt.close('all')
   
   
   # Systematics plot
   saturation_qfit = lnks.q_hst.max()*0.8
   
   typical_dispersion = (avg_pm['hst_gaia_pmra_%s_std'%(use_stat)] + avg_pm['hst_gaia_pmdec_%s_std'%(use_stat)]) / 2
   pmra_lims = [avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] - 5 * typical_dispersion, avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] + 5 * typical_dispersion]
   pmdec_lims = [avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] - 5 * typical_dispersion, avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] + 5 * typical_dispersion]
   

   # Astrometric plots
   
   id_std_hst_gaia = np.isfinite(lnks.xc_hst) & np.isfinite(lnks.yc_hst) & np.isfinite(lnks.relative_hst_gaia_pmra) & np.isfinite(lnks.relative_hst_gaia_pmdec)
   lnks_members = lnks.index.isin(table[table.use_for_alignment == True].index) & np.isfinite(lnks.xc_hst) & np.isfinite(lnks.yc_hst) & np.isfinite(lnks.relative_hst_gaia_pmra) & np.isfinite(lnks.relative_hst_gaia_pmdec)

   fig, axs = plt.subplots(2, 2, sharex = False, sharey = False, figsize = (10, 5))

   bin_xhst, mean_xhst_pmra_hst_gaia, std_xhst_pmra_hst_gaia, mean_xhst_pmra_error_hst_gaia = bin_errors(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members], lnks.relative_hst_gaia_pmra_error[lnks_members], n_bins = 10)
   bin_yhst, mean_yhst_pmra_hst_gaia, std_yhst_pmra_hst_gaia, mean_yhst_pmra_error_hst_gaia = bin_errors(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members], lnks.relative_hst_gaia_pmra_error[lnks_members], n_bins = 10)

   bin_xhst, mean_xhst_pmdec_hst_gaia, std_xhst_pmdec_hst_gaia, mean_xhst_pmdec_error_hst_gaia = bin_errors(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members], lnks.relative_hst_gaia_pmdec_error[lnks_members], n_bins = 10)
   bin_yhst, mean_yhst_pmdec_hst_gaia, std_yhst_pmdec_hst_gaia, mean_yhst_pmdec_error_hst_gaia = bin_errors(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members], lnks.relative_hst_gaia_pmdec_error[lnks_members], n_bins = 10)

   q_hst = axs[0,0].scatter(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[lnks_members], s = 1)
   axs[0,0].scatter(lnks.xc_hst[~lnks_members], lnks.relative_hst_gaia_pmra[~lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')

   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'r', label='mean')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] + std_xhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label=r'$\sigma$')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] - std_xhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] + mean_xhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label=r'$error$')
   axs[0,0].plot(bin_xhst, mean_xhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] - mean_xhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(lnks.xc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[lnks_members], s = 1)
   axs[0,1].scatter(lnks.xc_hst[~lnks_members], lnks.relative_hst_gaia_pmdec[~lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')

   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] + std_xhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] - std_xhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] + mean_xhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_xhst, mean_xhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] - mean_xhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[1,0].scatter(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmra[lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[lnks_members], s = 1)
   axs[1,0].scatter(lnks.yc_hst[~lnks_members], lnks.relative_hst_gaia_pmra[~lnks_members] + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')

   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] + std_yhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] - std_yhst_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] + mean_yhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[1,0].plot(bin_yhst, mean_yhst_pmra_hst_gaia + avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)] - mean_yhst_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[1,1].scatter(lnks.yc_hst[lnks_members], lnks.relative_hst_gaia_pmdec[lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[lnks_members], s = 1)
   axs[1,1].scatter(lnks.yc_hst[~lnks_members], lnks.relative_hst_gaia_pmdec[~lnks_members] + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], c = lnks.q_hst[~lnks_members], s = 1, alpha = 0.35)
   axs[1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')

   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] + std_yhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] - std_yhst_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] + mean_yhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[1,1].plot(bin_yhst, mean_yhst_pmdec_hst_gaia + avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)] - mean_yhst_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'x [pix]') for ax in axs[0,:]]
   [ax.set_xlabel(r'y [pix]') for ax in axs[1,:]]
   [ax.grid() for ax in axs.flatten()]
   
   axs[0,0].legend()


   # Set limits
   #print('pmra_lims', pmra_lims)
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_5+ext, bbox_inches='tight')

   
   plt.close('all')
   # Photometry plots

   id_std_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec']) & (table.use_for_alignment == True)
   id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_stat]) & (table.use_for_alignment == True)

   bin_mag, mean_mag_pmra_hst_gaia, std_mag_pmra_hst_gaia, mean_mag_pmra_error_hst_gaia = bin_errors(table.gmag[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)
   bin_mag, mean_mag_pmdec_hst_gaia, std_mag_pmdec_hst_gaia, mean_mag_pmdec_error_hst_gaia = bin_errors(table.gmag[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)

   # Find HST filters
   hst_filters = [col for col in table.columns if ('F' in col) & ('error' not in col) & ('std' not in col) & ('_mean' not in col)]
   hst_filters.sort()

   fig, axs = plt.subplots(len(hst_filters)+1, 2, sharex = False, sharey = False, figsize = (10.0, 3*len(hst_filters)+1))

   axs[0,0].scatter(table.gmag[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,0].scatter(table.gmag[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)
   
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
   
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r', label=r'mean')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia + std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label=r'$\sigma$')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia - std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia + mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label=r'error')
   axs[0,0].plot(bin_mag, mean_mag_pmra_hst_gaia - mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(table.gmag[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,1].scatter(table.gmag[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   for ii, cmd_filter in enumerate(hst_filters):
      hst_filter = table[cmd_filter].rename('%s'%cmd_filter.replace("_wmean","").replace("_mean",""))
      
      id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_stat]) & np.isfinite(hst_filter) & (table.use_for_alignment == True)
      
      bin_mag, mean_mag_pmra_hst_gaia, std_mag_pmra_hst_gaia, mean_mag_pmra_error_hst_gaia = bin_errors(hst_filter[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)
      bin_mag, mean_mag_pmdec_hst_gaia, std_mag_pmdec_hst_gaia, mean_mag_pmdec_error_hst_gaia = bin_errors(hst_filter[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)

      axs[ii+1,0].scatter(hst_filter[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia + std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia - std_mag_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia + mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_mag, mean_mag_pmra_hst_gaia - mean_mag_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,1].scatter(hst_filter[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,1].scatter(hst_filter[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - std_mag_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia + mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_mag, mean_mag_pmdec_hst_gaia - mean_mag_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,0].set_xlabel(hst_filter.name)
      axs[ii+1,1].set_xlabel(hst_filter.name)


   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'G') for ax in axs[0,:]]
   [ax.grid() for ax in axs.flatten()]
   
   axs[0,0].legend()
   
   # Set limits
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_6+ext, bbox_inches='tight')


   plt.close('all')

   # Color figures
   id_std_gaia = np.isfinite(table['pmra']) & np.isfinite(table['pmdec']) & (table.use_for_alignment == True)
   id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_stat]) & (table.use_for_alignment == True) & np.isfinite(table['bp_rp'])

   bin_color, mean_color_pmra_hst_gaia, std_color_pmra_hst_gaia, mean_color_pmra_error_hst_gaia = bin_errors(table.bp_rp[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)
   bin_color, mean_color_pmdec_hst_gaia, std_color_pmdec_hst_gaia, mean_color_pmdec_error_hst_gaia = bin_errors(table.bp_rp[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)

   fig, axs = plt.subplots(len(hst_filters)+1, 2, sharex = False, sharey = False, figsize = (10, 3*len(hst_filters)+1))

   axs[0,0].scatter(table.bp_rp[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,0].scatter(table.bp_rp[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)
   
   axs[0,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r', label = 'mean')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia + std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r', label = r'$\sigma$')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia - std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia + mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r', label = 'error')
   axs[0,0].plot(bin_color, mean_color_pmra_hst_gaia - mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')


   axs[0,1].scatter(table.bp_rp[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
   axs[0,1].scatter(table.bp_rp[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

   axs[0,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia + std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia - std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia + mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
   axs[0,1].plot(bin_color, mean_color_pmdec_hst_gaia - mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

   for ii, cmd_filter in enumerate(hst_filters):
      if int(re.findall(r'\d+', cmd_filter)[0]) <= 606:
         HST_Gaia_color = (table[cmd_filter]-table['gmag']).rename('%s - G'%cmd_filter.replace("_wmean","").replace("_mean",""))
      else:
         HST_Gaia_color = (table['gmag']-table[cmd_filter]).rename('G - %s'%cmd_filter.replace("_wmean","").replace("_mean",""))

      id_std_hst_gaia = np.isfinite(table['hst_gaia_pmra_%s'%use_stat]) & np.isfinite(table['hst_gaia_pmdec_%s'%use_stat]) & np.isfinite(HST_Gaia_color) & (table.use_for_alignment == True)

      bin_color, mean_color_pmra_hst_gaia, std_color_pmra_hst_gaia, mean_color_pmra_error_hst_gaia = bin_errors(HST_Gaia_color[id_std_hst_gaia], table['hst_gaia_pmra_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmra_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)
      bin_color, mean_color_pmdec_hst_gaia, std_color_pmdec_hst_gaia, mean_color_pmdec_error_hst_gaia = bin_errors(HST_Gaia_color[id_std_hst_gaia], table['hst_gaia_pmdec_%s'%use_stat][id_std_hst_gaia], table['hst_gaia_pmdec_%s_error'%use_stat][id_std_hst_gaia], n_bins = 10)

      axs[ii+1,0].scatter(HST_Gaia_color[table.use_for_alignment == False], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,0].scatter(HST_Gaia_color[table.use_for_alignment == True], table['hst_gaia_pmra_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,0].axhline(y = avg_pm['hst_gaia_pmra_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia + std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia - std_color_pmra_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia + mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,0].plot(bin_color, mean_color_pmra_hst_gaia - mean_color_pmra_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,1].scatter(HST_Gaia_color[table.use_for_alignment == False], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == False], c = table['q_hst_mean'][table.use_for_alignment == False], s = 1, alpha = 0.35)
      axs[ii+1,1].scatter(HST_Gaia_color[table.use_for_alignment == True], table['hst_gaia_pmdec_%s'%use_stat][table.use_for_alignment == True], c = table['q_hst_mean'][table.use_for_alignment == True], s = 1)

      axs[ii+1,1].axhline(y = avg_pm['hst_gaia_pmdec_%s_%s'%(use_stat, use_stat)], linestyle = '-', linewidth = 1, color = 'k')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia, linestyle = '-', linewidth = 1, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia + std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia - std_color_pmdec_hst_gaia, linestyle = '-', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia + mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')
      axs[ii+1,1].plot(bin_color, mean_color_pmdec_hst_gaia - mean_color_pmdec_error_hst_gaia, linestyle = '--', linewidth = 0.75, color = 'r')

      axs[ii+1,0].set_xlabel(HST_Gaia_color.name)
      axs[ii+1,1].set_xlabel(HST_Gaia_color.name)


   # Set titles
   [ax.set_ylabel(r'$\mu_{\alpha*}$ [m.a.s./yr.]') for ax in axs[:,0]]
   [ax.set_ylabel(r'$\mu_{\delta}$ [m.a.s./yr.]') for ax in axs[:,1]]
   [ax.set_xlabel(r'$G$') for ax in axs[0,:]]
   [ax.grid() for ax in axs.flatten()]
   
   # Set limits
   [ax.set_ylim(pmra_lims) for ax in axs[:,0]]
   [ax.set_ylim(pmdec_lims) for ax in axs[:,1]]

   plt.subplots_adjust(wspace=0.5, hspace=0.3, right = 0.8)
   
   # Colorbar
   cbar = fig.colorbar(q_hst, ax=axs.ravel().tolist(), ticks=[lnks.q_hst.min(), saturation_qfit, lnks.q_hst.max()], aspect = 40, pad = 0.05)
   cbar.ax.set_yticklabels([str(round(lnks.q_hst.min(), 1)), 'sat', str(round(lnks.q_hst.max(), 1))])
   cbar.set_label('qfit')
   cbar.ax.plot([0, 1], [saturation_qfit, saturation_qfit], 'k')

   plt.savefig(plot_name_7+ext, bbox_inches='tight')
   

def get_object_properties(args):
   """
   This routine will try to obtain all the required object properties from Simbad or from the user.
   """

   print('\n'+'-'*42)
   print("Commencing execution")
   print('-'*42)

   #Try to get object:
   if (args.ra is None) or (args.dec is None):
      try:
         from astroquery.simbad import Simbad
         #query the SIMBAD service
         import astropy.units as u
         from astropy.coordinates import SkyCoord

         customSimbad = Simbad()
         customSimbad.add_votable_fields('dim')

         object_table = customSimbad.query_object(args.name)
         
         object_name = str(object_table['MAIN_ID'][0]).replace("b'NAME ","'").replace("b' ","'")
         coo = SkyCoord(ra = object_table['RA'], dec = object_table['DEC'], unit=(u.hourangle, u.deg))

         args.ra = float(coo.ra.deg)
         args.dec = float(coo.dec.deg)

         #Try to get the search radius
         if all((args.search_radius == None, any((args.search_width == None, args.search_height == None)))):
            if (object_table['GALDIM_MAJAXIS'].mask == False):
               args.search_radius = max(np.round(float(2. * object_table['GALDIM_MAJAXIS'] / 60.), 2), 0.1)

      except:
         object_name = args.name
         if ((args.ra is None) or (args.dec is None)) and (args.quiet is False):
            print('\n')
            try:
               if (args.ra is None):
                  args.ra = float(input('R.A. not defined, please enter R.A. in degrees: '))
               if args.dec is None:
                  args.dec = float(input('Dec not defined, please enter Dec in degrees: '))
            except:
               print('No valid input. Float number required.')
               print('\nExiting now.\n')
               sys.exit(1)

         elif ((args.ra is None) or (args.dec is None)) and (args.quiet is True):
            print('GaiaJWST could not find the object coordinates. Please check that the name of the object is written correctly. You can also run GaiaHub deffining explictly the coordinates using the "--ra" and "--dec" options.')
            sys.exit(1)

   else:
      object_name = args.name

   if (args.search_radius is None) and (args.quiet is False):
      print('\n')
      args.search_radius = float(input('Search radius not defined, please enter the search radius in degrees (Press enter to adopt the default value of 0.25 deg): ') or 0.25)
   elif (args.search_radius is None) and (args.quiet is True):
      args.search_radius = 0.25

   if (args.search_height is None):
      try:
         args.search_height = 2.*args.search_radius
      except:
         args.search_height = 1.0
   if (args.search_width is None):
      try:
         args.search_width = np.abs(2.*args.search_radius/np.cos(np.deg2rad(args.dec)))
      except:
         args.search_width = 1.0

   setattr(args, 'area', args.search_height * args.search_width * np.abs(np.cos(np.deg2rad(args.dec))))

   if args.jwst_filters == ['any']:
      args.jwst_filters = ['F090W','F115W','F140W','F150W','F158W','F200W','F277W',\
                          'F356W','F380W','F430W', 'F444W','F480W', 'F560W', 'F770W', 'F1000W']
      #args.hst_filters = ['F555W','F606W','F775W','F814W','F850LP']
      # args.hst_filters = ['F658N','F814W','F435W','F606W','F625W','F850LP',\
      #                      'F775W','F475W','F555W','F467M','F225W',\
      #                      'F336W','F275W','F390W','F438W','F220W','F330W']

   name_coo = 'ra_%.3f_dec_%.3f_r_%.2f'%(args.ra, args.dec, args.search_radius)

   if args.name is not None:
      args.name = args.name.replace(" ", "_")
      args.base_file_name = args.name+'_'+name_coo
   else:
      args.name = name_coo
      args.base_file_name = name_coo
   
   args.exec_path = './tmp_%s'%args.name
   
   #The script creates directories and set files names
   args.base_path = './%s/'%(args.name)
   args.JWST_path = args.base_path+'JWST/'
   args.Gaia_path = args.base_path+'Gaia/'
   args.Gaia_ind_queries_path = args.Gaia_path+'individual_queries/'

   options = '_%s'%args.use_stat
   if args.use_members:
      options += '_members'
   if args.rewind_stars:
      options += '_rewind'

   args.GaiaJWST_output = args.base_path+'Results%s/'%options
   
   args.used_JWST_obs_table_filename = args.GaiaJWST_output + args.base_file_name+'_used_JWST_images.csv'
   args.JWST_Gaia_table_filename = args.GaiaJWST_output + args.base_file_name+'_%s.csv'%args.use_stat
   args.logfile = args.GaiaJWST_output + args.base_file_name+'_%s.log'%args.use_stat
   args.queries = args.GaiaJWST_output + args.base_file_name+'_queries.log'
   
   args.Gaia_clean_table_filename = args.Gaia_path + args.base_file_name+'_gaia.csv'
   args.JWST_obs_table_filename = args.JWST_path + args.base_file_name+'_obs.csv'
   args.JWST_data_table_products_filename = args.JWST_path + args.base_file_name+'_data_products.csv'
   args.lnks_summary_filename = args.JWST_path + args.base_file_name+'_lnks_summary.csv'

   args.date_second_epoch = Time('%4i-%02i-%02iT00:00:00.000'%(args.date_second_epoch[2], args.date_second_epoch[0], args.date_second_epoch[1])).mjd
   args.date_reference_second_epoch = Time(args.date_reference_second_epoch)
   
   # Set up the number of processors:
   args.n_processes = use_processors(args.n_processes)
   
   print('\n')
   print('-'*42)
   print('Search information')
   print('-'*42)
   print('- Object name:', object_name)
   print('- (ra, dec) = (%s, %s) deg.'%(round(args.ra, 5), round(args.dec, 5)))
   print('- Search radius = %s deg.'%args.search_radius)
   print('-'*42+'\n')

   return args


def str2bool(v):
   """
   This routine converts ascii input to boolean.
   """
 
   if v.lower() in ('yes', 'true', 't', 'y'):
      return True
   elif v.lower() in ('no', 'false', 'f', 'n'):
      return False
   else:
      raise argparse.ArgumentTypeError('Boolean value expected.')
   

def get_real_error(table):
   """
   This routine calculates the excess of error that should be added to the listed Gaia errors. The lines are based on Fabricius 2021. Figure 21.
   """

   p5 = table.astrometric_params_solved == 31 # 5p parameters solved Brown et al. (2020)
   
   table.loc[:, ['parallax_error_old', 'pmra_error_old', 'pmdec_error_old']] = table.loc[:, ['parallax_error', 'pmra_error', 'pmdec_error']].values
   
   table.loc[p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']] = table.loc[p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']].values * 1.05
   table.loc[~p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']] = table.loc[~p5, ['ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error']].values * 1.22

   return table

