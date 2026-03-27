import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def cat2hpx(ra, dec,texp, nside):
    print('Num Obs: ', len(ra))
    npix = hp.nside2npix(nside)

    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)
    #print('Length of indices', len(indices))
    idx, counts = np.unique(indices, return_counts=True)
    #print('Length of idx', len(idx))
    #print('Length of texp', len(texp))

    fhSum = np.bincount(indices, weights=texp)
    #print('Length of fhSum', len(fhSum[fhSum>0]))
    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=np.float64)
    hpx_map[idx] = fhSum[fhSum>0]
    hpx_map[hpx_map==0] = np.nan

    return hpx_map

qvp = pd.read_csv("./visit_plans/visits_SELFIE593.txt", delim_whitespace=True, comment='#', 
                    names=['id_tile', 'ra', 'dec', 'pos', 'isky', 'texp', 'texp_ob', 'tob_len', 'irank', 'ntile'])
print(qvp.head())

ra = qvp['ra']
dec = qvp['dec']
texp = qvp['texp']
nside = 16
hpx_map = cat2hpx(ra, dec, texp, nside)

# --- Cartesian Projection ---
xsize = 800
ysize = int(xsize / 2)
proj = hp.projector.CartesianProj(xsize=xsize, ysize=ysize)

# Align the HEALPix map sphere orientation (nest=True to match cat2hpx)
npix = hp.nside2npix(nside)
vec = hp.pix2vec(nside, np.arange(npix), nest=True)
r = hp.Rotator(rot=[180, 0, 0], deg=True)
vec_rot = r(vec)
new_pix = hp.vec2pix(nside, *vec_rot, nest=True)

# Rearrange indices for the correct rotation
hp_map_rotated = hpx_map[new_pix]

# Map the 1D Array to 2D Image Space
vec2pix_func = lambda x, y, z: hp.vec2pix(nside, x, y, z, nest=True)
image_array = proj.projmap(hp_map_rotated, vec2pix_func)

from matplotlib.colors import LogNorm

# Plot the 2D projected image
plt.figure(figsize=(12, 6))
# Setting extent mathematically: X from 360 down to 0, Y from -90 up to 90

# Mask negative or zero values safely so log10 doesn't throw a math domain error
masked_image = np.ma.masked_invalid(image_array)
masked_image = np.ma.masked_less_equal(masked_image, 0)

plt.imshow(masked_image, origin='lower', extent=[360, 0, -90, 90], cmap='magma', norm=LogNorm())
plt.colorbar(label='texp total (log scale)')
plt.title('QVP Visit Plan (Cartesian Projection)')
plt.xlabel('RA (degrees)')
plt.ylabel('Dec (degrees)')
plt.grid(color='white', linestyle='--', alpha=0.3)
plt.show()
