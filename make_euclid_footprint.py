"""
Convert the Euclid DR1 HEALPix footprint mask into a lightweight polygon JSON
for visual overlay in appOverlay.py. Euclid is NOT an input to the LTS.

The input map is a full-sky NESTED HEALPix map (NSIDE=8192, ~6GB uncompressed),
so it is stream-decompressed and downsampled on the fly: in NESTED ordering each
coarse pixel is a contiguous block of (nside_in/nside_out)**2 fine pixels.

Usage:
    python make_euclid_footprint.py [--nside 128] [--plot check.png]
"""
import argparse
import gzip
import json

import healpy as hp
import numpy as np
from astropy.io import fits
from shapely.affinity import translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def read_header(path):
    with fits.open(path) as hdul:
        hdu = hdul[1]
        hdr = hdu.header
        if hdr.get('ORDERING', '').strip().upper() != 'NESTED':
            raise ValueError("Streaming downsample requires NESTED ordering")
        if hdr.get('OBJECT', 'FULLSKY').strip().upper() != 'FULLSKY':
            raise ValueError("Only FULLSKY (implicit index) maps are supported")
        if hdr['TFIELDS'] != 1 or not hdr['TFORM1'].strip().endswith('D'):
            raise ValueError(f"Unexpected column format {hdr['TFORM1']}")
        return hdr['NSIDE'], hdu.fileinfo()['datLoc']


def coverage_fraction(path, nside_in, nside_out, data_offset):
    """Stream the map and return the fraction of good fine pixels per coarse pixel."""
    ratio = (nside_in // nside_out) ** 2
    npix_out = hp.nside2npix(nside_out)
    frac = np.empty(npix_out, dtype=np.float32)
    chunk_bytes = ratio * 2048 * 8
    filled = 0
    with gzip.open(path, 'rb') as f:
        f.seek(data_offset)
        while filled < npix_out:
            buf = f.read(chunk_bytes)
            if not buf:
                break
            v = np.frombuffer(buf, dtype='>f8')
            good = np.isfinite(v) & (v > 0) & (v > -1e30)
            block = good.reshape(-1, ratio).mean(axis=1)
            frac[filled:filled + len(block)] = block
            filled += len(block)
    if filled != npix_out:
        raise ValueError(f"Read {filled} coarse pixels, expected {npix_out}")
    return frac


def pixel_polygons(nside, pix):
    """Shapely polygons (RA/Dec degrees) for each pixel, unwrapped across RA=0."""
    corners = hp.boundaries(nside, np.asarray(pix), step=2, nest=True).reshape(len(pix), 3, -1)
    polys = []
    for c in corners:
        ra, dec = hp.vec2dir(c, lonlat=True)
        ra = np.mod(ra, 360.0)
        if ra.max() - ra.min() > 180:
            ra = np.where(ra < 180, ra + 360, ra)
        polys.append(Polygon(zip(ra, dec)))
    return polys


def split_at_wrap(geom):
    """Split geometry at RA=360 so every piece lies within [0, 360]."""
    main = geom.intersection(box(0, -90, 360, 90))
    wrapped = translate(geom.intersection(box(360, -90, 720, 90)), xoff=-360)
    return unary_union([main, wrapped])


def iter_polygons(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    return [g for g in getattr(geom, 'geoms', []) if g.geom_type == 'Polygon']


def sky_area(poly):
    """Approximate spherical area (deg^2) of an RA/Dec polygon."""
    return poly.area * np.cos(np.deg2rad(poly.centroid.y))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='euclidFields/cgv_map_dr1input_o13.fits.gz')
    parser.add_argument('--output', default='euclidFields/euclid_dr1_footprint.json')
    parser.add_argument('--nside', type=int, default=128, help="Output NSIDE (default 128)")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Min fraction of covered fine pixels for a coarse pixel to count")
    parser.add_argument('--min-area', type=float, default=1.0, help="Drop regions smaller than this (deg^2)")
    parser.add_argument('--simplify', type=float, default=0.05, help="Simplification tolerance (deg)")
    parser.add_argument('--plot', default=None, help="Optional PNG path for a quick-look check plot")
    args = parser.parse_args()

    nside_in, data_offset = read_header(args.input)
    if nside_in % args.nside or not hp.isnsideok(args.nside):
        raise ValueError(f"Output NSIDE {args.nside} must be a power of 2 dividing {nside_in}")

    print(f"Streaming {args.input} (NSIDE={nside_in}) -> NSIDE={args.nside} ...")
    frac = coverage_fraction(args.input, nside_in, args.nside, data_offset)
    pix = np.where(frac > args.threshold)[0]
    pix_area = len(pix) * hp.nside2pixarea(args.nside, degrees=True)
    print(f"{len(pix)} pixels above threshold {args.threshold} ({pix_area:.1f} deg^2)")

    merged = split_at_wrap(unary_union(pixel_polygons(args.nside, pix)))

    regions = []
    for g in iter_polygons(merged):
        g = Polygon(g.exterior).simplify(args.simplify, preserve_topology=True)
        if sky_area(g) >= args.min_area:
            regions.append(g)
    regions.sort(key=sky_area, reverse=True)

    areas = []
    for n, g in enumerate(regions, start=1):
        ra, dec = g.exterior.coords.xy
        areas.append({
            "name": f"Euclid DR1 region {n}",
            "type": "polygon",
            "RA": [round(float(x), 4) for x in ra],
            "Dec": [round(float(y), 4) for y in dec],
            "t_frac": 0.0,
        })
        print(f"  region {n}: {sky_area(g):8.1f} deg^2  RA {min(ra):6.1f}-{max(ra):6.1f}  "
              f"Dec {min(dec):6.1f}-{max(dec):6.1f}  ({len(ra)} vertices)")
    total = sum(sky_area(g) for g in regions)
    print(f"{len(regions)} polygons, total {total:.1f} deg^2 (pixel area {pix_area:.1f} deg^2)")

    out = {
        "survey": "Euclid DR1",
        "author": "make_euclid_footprint.py",
        "scienceJustification": (
            f"Visualisation only - not an LTS input. Generated from {args.input} "
            f"at NSIDE={args.nside}, coverage threshold {args.threshold}, holes filled."
        ),
        "year1Areas": areas,
    }
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ra_pix, dec_pix = hp.pix2ang(args.nside, pix, nest=True, lonlat=True)
        ax.scatter(ra_pix, dec_pix, s=1, c='lightgrey', label='NSIDE pixels')
        for a in areas:
            ax.plot(a['RA'], a['Dec'], '-', lw=1)
        ax.set_xlim(360, 0)
        ax.set_ylim(-90, 90)
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title('Euclid DR1 footprint')
        fig.savefig(args.plot, dpi=120, bbox_inches='tight')
        print(f"Wrote {args.plot}")


if __name__ == '__main__':
    main()
