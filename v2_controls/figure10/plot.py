#!/usr/bin/env python3
"""Portable paper-style Figure 10 from source-derived pooled.csv only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, help='Default: data directory')
    args = parser.parse_args()
    config = json.loads((args.data_dir / 'config.json').read_text())
    with (args.data_dir / 'pooled.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    values = {(r['row_id'], r['condition']): float(r['rate']) for r in rows}
    expected = {(r['row_id'], c) for r in config['rows'] for c in config['columns']}
    if len(rows) != 65 or set(values) != expected:
        raise ValueError('Expected exactly 13 rows by 5 conditions')
    if any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError('Rate outside [0, 1]')
    # Original, portable approximation of the paper design. No image-derived
    # numerical inputs and no platform plotting/style dependencies.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': '#181818',
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=config['render']['figsize_inches'], facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    cmap = LinearSegmentedColormap.from_list('paper_white_red',
            [(0.0, '#ffffff'), (0.25, '#fae5e1'), (0.5, '#f2c6be'), (0.75, '#db8074'), (1.0, '#b83326')])
    centers_x = [0.446, 0.567, 0.688, 0.809, 0.930]
    headings = ['No\nsteering', 'Random', 'Fear', 'Sadness', 'Pain']
    for x, label in zip(centers_x, headings):
        ax.text(x, 0.914, label, ha='center', va='center', fontsize=13.3, fontweight='bold', linespacing=1.05)
    ax.text(0.010, 0.986, 'Destructive choices', ha='left', va='top', fontsize=15.6, fontweight='bold')
    ax.text(0.010, 0.386, 'Non-destructive choices', ha='left', va='top', fontsize=15.6, fontweight='bold')
    for i, row in enumerate(config['rows']):
        y = 0.8615 - 0.0535 * i if i < 9 else 0.322 - 0.064 * (i - 9)
        height = 0.0475 if i < 9 else 0.0555
        ax.text(0.381, y, row['label'], ha='right', va='center', fontsize=12.7, linespacing=1.16)
        for x, condition in zip(centers_x, config['columns']):
            value = values[(row['row_id'], condition)]
            ax.add_patch(Rectangle((x - 0.057, y - height / 2), 0.114, height,
                                   facecolor=cmap(value), edgecolor='none'))
            ax.text(x, y, f'{value * 100:.0f}%', ha='center', va='center', fontsize=15.3,
                    color='white' if value >= 0.70 else '#181818',
                    fontweight='bold' if condition == 'pain' else 'normal')
    # Unlike the abbreviated paper footer, explicitly retain the 72B dose.
    ax.text(0.410, 0.020,
            '% of first choices selecting the first option · Qwen 2.5 32B unless noted · coefficient 1.0 (72B: 1.25)',
            ha='left', va='center', fontsize=10.0, color='#6b7280')
    out = args.output_dir or args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / config['render']['png'], dpi=config['render']['dpi'], facecolor='white')
    fig.savefig(out / config['render']['pdf'], facecolor='white', metadata={'Title': 'Figure 10: choice controls', 'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    print('Rendered figure10.png and figure10.pdf from 65 source-derived rates; fixed color domain [0,1].')


if __name__ == '__main__':
    main()
