#!/usr/bin/env python3
"""
Generate a file with lines:
GOOGLE_API_KEY_1=sss
...
GOOGLE_API_KEY_n=sss

Usage:
  python generate_keys.py 5            # writes to google_api_keys.env
  python generate_keys.py 5 -o my.env  # custom output file
"""

import argparse
import sys

API_VALUE = "sss"  # hardcoded as requested

def main():
    parser = argparse.ArgumentParser(description="Write GOOGLE_API_KEY_i=sss lines to a file.")
    parser.add_argument("n", type=int, help="Number of keys to write (positive integer).")
    parser.add_argument("-o", "--out", default="google_api_keys.env",
                        help="Output filename (default: google_api_keys.env)")
    args = parser.parse_args()

    if args.n < 1:
        print("n must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(args.out, "w", encoding="utf-8") as f:
            for i in range(1, args.n + 1):
                f.write(f"GOOGLE_API_KEY_{i}={API_VALUE}\n")
        print(f"Wrote {args.n} keys to {args.out}")
    except OSError as e:
        print(f"Failed to write file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()