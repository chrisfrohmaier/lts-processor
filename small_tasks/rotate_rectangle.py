#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

def main():
    parser = argparse.ArgumentParser(
        description="Calculate the corner coordinates of a rotated rectangle on the sky and output in LTS polygon JSON format."
    )
    # Coordinate and dimension arguments
    parser.add_argument("--ra", type=float, required=True, help="Center RA in degrees")
    parser.add_argument("--dec", type=float, required=True, help="Center Dec in degrees")
    parser.add_argument("--width", type=float, required=True, help="Width of the rectangle in degrees")
    parser.add_argument("--height", type=float, required=True, help="Height of the rectangle in degrees")
    parser.add_argument(
        "--rotation", 
        type=float, 
        default=0.0, 
        help="Rotation angle in degrees. 0 means North is straight up (along height axis). Positive values rotate the rectangle counter-clockwise (East to North in sky offset coordinates)."
    )

    # LTS polygon metadata arguments
    parser.add_argument("--year", type=int, default=1, help="Default year for the LTS polygon (default: 1)")
    parser.add_argument("--tfrac", type=float, default=1.0, help="t_frac value for the polygon (default: 1.0)")
    parser.add_argument("--name", type=str, default="rotated_rectangle", help="Name of the polygon (default: rotated_rectangle)")
    parser.add_argument("--survey", type=str, default="S00", help="Survey name in the output JSON (default: S00)")
    parser.add_argument("--author", type=str, default="Your Name", help="Author name in the output JSON (default: Your Name)")
    parser.add_argument("--science-justification", type=str, default="", help="Science justification in the output JSON (default: '')")

    args = parser.parse_args()

    # Define the center coordinate
    center = SkyCoord(ra=args.ra * u.deg, dec=args.dec * u.deg, frame='icrs')

    # Define the unrotated offsets for the 4 corners
    # The order of corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    # Top-Left: West (-) and North (+)
    # Top-Right: East (+) and North (+)
    # Bottom-Right: East (+) and South (-)
    # Bottom-Left: West (-) and South (-)
    w_half = args.width / 2.0
    h_half = args.height / 2.0

    base_corners = [
        (-w_half, h_half),  # Top-Left
        (w_half, h_half),   # Top-Right
        (w_half, -h_half),  # Bottom-Right
        (-w_half, -h_half)  # Bottom-Left
    ]

    # Convert rotation angle to radians and construct rotation matrix
    rot_rad = np.radians(args.rotation)
    cos_r = np.cos(rot_rad)
    sin_r = np.sin(rot_rad)

    # Store RA and Dec list for the output JSON
    ra_coords = []
    dec_coords = []

    # Write summary info to stderr so it doesn't pollute piped stdout JSON
    print(f"Rectangle Center: RA = {args.ra:.6f}°, Dec = {args.dec:.6f}°", file=sys.stderr)
    print(f"Dimensions: Width = {args.width:.6f}°, Height = {args.height:.6f}°", file=sys.stderr)
    print(f"Rotation: {args.rotation:.6f}°", file=sys.stderr)
    print("-" * 50, file=sys.stderr)

    labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
    for label, (x, y) in zip(labels, base_corners):
        # Apply rotation matrix
        lon_offset = x * cos_r - y * sin_r
        lat_offset = x * sin_r + y * cos_r

        # Calculate absolute coordinate on the sphere
        corner_coord = SkyCoord(
            lon=lon_offset * u.deg,
            lat=lat_offset * u.deg,
            frame=center.skyoffset_frame()
        ).transform_to('icrs')

        ra_val = float(corner_coord.ra.deg)
        dec_val = float(corner_coord.dec.deg)
        ra_coords.append(round(ra_val, 6))
        dec_coords.append(round(dec_val, 6))

        print(f"{label:12s} : RA = {ra_val:10.6f}°, Dec = {dec_val:10.6f}°", file=sys.stderr)

    # Build the LTS JSON format structure
    lts_json = {
        "survey": args.survey,
        "scienceJustification": args.science_justification,
        "author": args.author,
        "year1Areas": [
            {
                "name": args.name,
                "type": "polygon",
                "RA": ra_coords,
                "Dec": dec_coords,
                "t_frac": args.tfrac,
                "year": args.year
            }
        ]
    }

    # Print the JSON to stdout
    print(json.dumps(lts_json, indent=2))

if __name__ == "__main__":
    main()
