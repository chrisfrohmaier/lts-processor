import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from astropy.time import Time
from datetime import datetime, time

st.title("HEALPix Survey Strategies")

# NSIDE determines the resolution of the HEALPix map
nside_options = [16, 32, 64, 128, 256, 512]
NSIDE = st.select_slider("Select Map Resolution (NSIDE)", options=nside_options, value=16)

@st.cache_data
def load_csv(csv_path):
    """
    Loads data from the specified CSV file.
    """
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None


def cat2hpx_DateDiff(ra, dec, nside):
    #print('Num Obs: ', len(ra))
    npix = hp.nside2npix(nside)

    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi, nest=True)

    idx, counts = np.unique(indices, return_counts=True)


    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=np.float64)
    hpx_map[idx] = counts
    hpx_map[hpx_map==0] = np.nan


    return hpx_map

csv_files = glob.glob(os.path.join('.', 'strategies', '*.csv'))
csv_file_names = [os.path.basename(f) for f in sorted(csv_files)]

if not csv_files:
    st.error("No CSV files found in ./strategies folder.")
    df = None
else:
    selected_file_name = st.selectbox("Select Strategy CSV", csv_file_names)
    selected_csv = os.path.join('.', 'strategies', selected_file_name)
    df = load_csv(selected_csv)

xsize = 800
ysize = xsize // 2
longitude = np.linspace(360,0, xsize)
latitude = np.linspace(-90, 90, ysize)
proj = hp.projector.CartesianProj(xsize=xsize, ysize=ysize)
npix = hp.nside2npix(NSIDE)
vec = hp.pix2vec(NSIDE, np.arange(npix), nest=True)
r = hp.Rotator(rot=[180, 0, 0], deg=True)
vec_rot = r(vec)
new_pix = hp.vec2pix(NSIDE, *vec_rot, nest=True)

if df is not None and not df.empty:
    min_mjd = float(df['observationStartMJD'].min())
    
    # Convert the lowest MJD to a python date to use as the default in the calendar
    try:
        min_date = Time(min_mjd, format='mjd').to_datetime().date()
    except Exception:
        min_date = datetime.today().date()
        
    # Calendar date selector
    selected_date = st.date_input("Start Date", value=min_date)
    
    # Convert calendar date at midnight to MJD
    dt_midnight = datetime.combine(selected_date, time.min)
    start_date = Time(dt_midnight).mjd
    
    for i in range(5):
        year_start = start_date + i * 365
        year_end = start_date + (i + 1) * 365
        
        df_year = df[(df['observationStartMJD'] >= year_start) & (df['observationStartMJD'] < year_end)]
        
        ra_col = 'fieldRa' if 'fieldRa' in df_year.columns else 'fieldRA'
        ra = df_year[ra_col]
        dec = df_year['fieldDec']
        
        hp_map_standard = cat2hpx_DateDiff(ra, dec, NSIDE)

        hp_map = hp_map_standard[new_pix]

        vec2pix_func = lambda x, y, z: hp.vec2pix(NSIDE, x, y, z, nest=True)
        image_array = proj.projmap(hp_map, vec2pix_func)
        
        # Determine min and max for the slider
        valid_vals = image_array[~np.isnan(image_array)]
        if len(valid_vals) > 0:
            min_val = int(np.floor(np.min(valid_vals)))
            max_val = int(np.ceil(np.max(valid_vals)))
        else:
            min_val, max_val = 0, 1
            
        if min_val == max_val:
            max_val = min_val + 1
            
        threshold = st.slider(f"Threshold for Red Overlay (Year {i+1})", 
                              min_value=min_val, 
                              max_value=max_val, 
                              value=min_val,
                              step=1,
                              key=f"slider_yr_{i}")
        
        invert_red = st.checkbox(f"Invert Red Overlay (Above Threshold) (Year {i+1})", key=f"invert_yr_{i}")
        
        layout = go.Layout(
        autosize=False,
        width=800, 
        height=600,
        title=f"Projected NESTED HEALPix Map Year {i+1} (NSIDE={NSIDE})",
        xaxis=dict(
            title='R.A.',

        ),
        yaxis=dict(
            title='Declination',

        ))

        fig = go.Figure()
        
        # Base Heatmap
        fig.add_trace(go.Heatmap(
            z=image_array,
            x=longitude,
            y=latitude,
            colorscale='Viridis',
            zmin=800,
            zmax=max_val
        ))
        
        # Highlight Mask (Red)
        if invert_red:
            condition = (image_array > threshold) & (~np.isnan(image_array))
        else:
            condition = (image_array <= threshold) & (~np.isnan(image_array))
            
        mask_array = np.where(condition, 1, np.nan)
        fig.add_trace(go.Heatmap(
            z=mask_array,
            x=longitude,
            y=latitude,
            colorscale=[[0, 'red'], [1, 'red']],
            showscale=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            yaxis_scaleanchor="x"
        )
        # fig = px.imshow(
        #     image_array,
        #     origin='lower', # Matches astronomical coordinate convention
        #     labels={'color': 'Intensity'},
        #     title=f"Projected NESTED HEALPix Map Year {i+1} (NSIDE={NSIDE})",
        #     color_continuous_scale='Viridis'
        # )

        # Optional: Clean up axes to look like a map
        #fig.update_xaxes(showticklabels=False)
        #fig.update_yaxes(showticklabels=False)
        fig['layout']['xaxis']['autorange'] = "reversed"
        fig.update_layout(yaxis_range=[-90,30])

        st.plotly_chart(fig, use_container_width=True)
