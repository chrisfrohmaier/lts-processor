import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os
import glob
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="LSST Observation Density Visualizer", layout="wide")

st.title("LSST Observation Density Visualizer")

# Function to load data
@st.cache_data
def load_data(db_path):
    """
    Loads the 'observations' table from the specified sqlite database.
    """
    if not os.path.exists(db_path):
        st.error(f"File not found: {db_path}")
        return None
    
    try:
        engine = create_engine(f'sqlite:///{db_path}')
        # We limit to 1000 rows first to verify structure, then we can load all.
        # For production use with 700MB+ files, consider checking available RAM or loading only necessary columns.
        # For now, following user request to load table.
        df = pd.read_sql("SELECT * FROM observations", engine)
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# Sidebar for file selection
st.sidebar.header("Configuration")

# Find .db files in the current directory
# Note: In a real deployment, you might want to specify a data directory to scan.
# Here we scan the directory where app.py is running.
db_files = glob.glob("*.db")

if not db_files:
    st.warning("No .db files found in the current directory.")
else:
    selected_db = st.sidebar.selectbox("Select LSST Database File", db_files)

    if selected_db:
        st.write(f"Loading data from: **{selected_db}**")
        
        with st.spinner('Loading data... This may take a while for large files.'):
            df = load_data(selected_db)
        
        if df is not None:
            st.success(f"Loaded {len(df)} observations.")
            
            # Display raw data
            st.subheader("Raw Data Preview")
            st.dataframe(df.head())
            
            # HEALPix Visualization
            st.subheader("Observation Density (HEALPix)")

            import healpy as hp
            import numpy as np
            
            if 'fieldRA' in df.columns and 'fieldDec' in df.columns:
                # User selection for NSIDE
                nside = st.sidebar.select_slider("HEALPix NSIDE", options=[16, 32, 64, 128], value=64)
                npix = hp.nside2npix(nside)
                
                # Convert to radians and theta/phi
                # fieldRA is usually in degrees, fieldDec in degrees
                ra_rad = np.radians(df['fieldRA'])
                dec_rad = np.radians(df['fieldDec'])
                
                # theta = co-latitude (0 at North Pole, pi at South Pole) => 90 - dec
                theta = np.pi/2 - dec_rad
                phi = ra_rad
                
                # Get pixel indices
                pix_indices = hp.ang2pix(nside, theta, phi)
                
                # Bin counts
                m = np.bincount(pix_indices, minlength=npix)
                
                # Plot
                fig = plt.figure(figsize=(12, 8))
                hp.mollview(m, title=f"Observation Counts (NSIDE={nside})", fig=fig, unit="Counts")
                st.pyplot(fig)
            else:
                st.warning("Columns `fieldRA` and `fieldDec` not found. Cannot create HEALPix map.")

