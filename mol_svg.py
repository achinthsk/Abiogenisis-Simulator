"""
mol_svg.py — Pure-Python 2D molecule SVG renderer.

Uses RDKit only for SMILES parsing and 2D coordinate generation — both of which
are part of the core C++ library and require NO X11/libXrender. The actual drawing
is done entirely in Python string manipulation, producing a self-contained SVG.

Public entry point:
    smiles_to_svg(smiles: str, width=220, height=160) -> str | None
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

# Element colour palette (dark-background friendly)
ATOM_COLORS = {
    'C':  '#d0d4e8',   # light grey — de-emphasised so bonds read first
    'N':  '#7aa8f0',   # blue
    'O':  '#f07a7a',   # red
    'S':  '#f0d07a',   # yellow
    'P':  '#f0a86a',   # orange
    'F':  '#7af0c8',   # cyan
    'Cl': '#7af0c8',
    'Br': '#d0895a',
    'I':  '#b07ae0',
    'H':  '#606478',   # very muted
}
DEFAULT_ATOM_COLOR = '#c8ccdc'
BOND_COLOR         = '#9aa0b8'
BACKGROUND         = '#0a0c10'


def _scale_coords(positions: list[tuple[float, float]], w: int, h: int,
                  pad: int = 22) -> list[tuple[float, float]]:
    """Scale and centre a list of (x, y) positions into the viewport."""
    if not positions:
        return []
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    scale = min((w - 2 * pad) / span_x, (h - 2 * pad) / span_y)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    return [
        ((x - cx) * scale + w / 2, -(y - cy) * scale + h / 2)
        for x, y in positions
    ]


def _bond_order_lines(x1: float, y1: float, x2: float, y2: float,
                      bond_type: int) -> list[str]:
    """Return SVG <line> elements for a bond. bond_type: 1=single,2=double,3=triple."""
    import math
    lines = []
    color = BOND_COLOR

    if bond_type == 1:
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>'
        )
    elif bond_type >= 2:
        # Perpendicular offset for the second (and third) line
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length, dx / length   # perpendicular unit vector
        off = 2.0 if bond_type == 2 else 2.5

        # First line (centre or offset)
        if bond_type == 3:
            for d in (-off, 0.0, off):
                lines.append(
                    f'<line x1="{x1+nx*d:.2f}" y1="{y1+ny*d:.2f}" '
                    f'x2="{x2+nx*d:.2f}" y2="{y2+ny*d:.2f}" '
                    f'stroke="{color}" stroke-width="1.3" stroke-linecap="round"/>'
                )
        else:  # double
            for d in (-off / 2, off / 2):
                lines.append(
                    f'<line x1="{x1+nx*d:.2f}" y1="{y1+ny*d:.2f}" '
                    f'x2="{x2+nx*d:.2f}" y2="{y2+ny*d:.2f}" '
                    f'stroke="{color}" stroke-width="1.4" stroke-linecap="round"/>'
                )
    return lines


def smiles_to_svg(smiles: str, width: int = 220, height: int = 160) -> str | None:
    """Generate a self-contained SVG string from a SMILES.
    Returns None if the SMILES is invalid.
    """
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Add explicit H coords only for terminal H on heteroatoms (cleaner diagrams)
    mol = Chem.AddHs(mol, onlyOnAtoms=[
        a.GetIdx() for a in mol.GetAtoms()
        if a.GetAtomicNum() != 6 and a.GetTotalNumHs() > 0
    ])

    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # Collect atom positions and symbols (skip most H for clarity)
    atoms = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        pos = conf.GetAtomPosition(atom.GetIdx())
        # Show H labels only when heteroatom-bound
        if sym == 'H':
            # Check if bonded to non-carbon
            for bond in atom.GetBonds():
                other = bond.GetOtherAtom(atom)
                if other.GetAtomicNum() != 6:
                    atoms.append((atom.GetIdx(), sym, pos.x, pos.y))
            # Always skip bare H2 molecule lone H atoms
            continue
        atoms.append((atom.GetIdx(), sym, pos.x, pos.y))

    if not atoms:
        # Fallback: show H atoms if nothing else
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            pos = conf.GetAtomPosition(atom.GetIdx())
            atoms.append((atom.GetIdx(), sym, pos.x, pos.y))

    idx_to_scaled = {}
    raw_positions = [(a[2], a[3]) for a in atoms]
    scaled = _scale_coords(raw_positions, width, height)
    for i, (idx, sym, rx, ry) in enumerate(atoms):
        idx_to_scaled[idx] = scaled[i]

    # Also need positions for ALL atoms for bond drawing (including hidden H)
    all_scaled: dict[int, tuple[float, float]] = {}
    all_raw = []
    all_idxs = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        all_raw.append((pos.x, pos.y))
        all_idxs.append(atom.GetIdx())
    all_sc = _scale_coords(all_raw, width, height)
    for atom_idx, sc in zip(all_idxs, all_sc):
        all_scaled[atom_idx] = sc

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" rx="8"/>')

    # Draw bonds
    from rdkit.Chem import rdchem
    BOND_TYPE_MAP = {
        rdchem.BondType.SINGLE:    1,
        rdchem.BondType.DOUBLE:    2,
        rdchem.BondType.TRIPLE:    3,
        rdchem.BondType.AROMATIC:  1,  # treat aromatic as single (dashes handled below)
    }
    aromatic_bonds = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i not in all_scaled or j not in all_scaled:
            continue
        x1, y1 = all_scaled[i]
        x2, y2 = all_scaled[j]
        btype = bond.GetBondType()
        order = BOND_TYPE_MAP.get(btype, 1)

        if btype == rdchem.BondType.AROMATIC:
            aromatic_bonds.append((i, j, x1, y1, x2, y2))
        parts += _bond_order_lines(x1, y1, x2, y2, order)

    # Draw dashed second lines for aromatic bonds
    for i, j, x1, y1, x2, y2 in aromatic_bonds:
        import math
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1
        nx, ny = -dy / length * 2.0, dx / length * 2.0
        parts.append(
            f'<line x1="{x1+nx:.2f}" y1="{y1+ny:.2f}" '
            f'x2="{x2+nx:.2f}" y2="{y2+ny:.2f}" '
            f'stroke="{BOND_COLOR}" stroke-width="1.2" stroke-dasharray="2,2" '
            f'stroke-linecap="round" opacity="0.6"/>'
        )

    # Draw atom labels (omit C unless it has no bonds or is explicit)
    FONT_SIZE = 10
    for (idx, sym, rx, ry), (sx, sy) in zip(atoms, scaled):
        color = ATOM_COLORS.get(sym, DEFAULT_ATOM_COLOR)
        atom = mol.GetAtomWithIdx(idx)
        degree = atom.GetDegree()

        # Skip carbon labels unless isolated (degree 0) or explicit
        if sym == 'C' and degree > 0:
            # Draw small filled circle at carbon junctions instead
            parts.append(
                f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="1.5" fill="{BOND_COLOR}" opacity="0.5"/>'
            )
            continue

        # White halo for readability
        parts.append(
            f'<text x="{sx:.2f}" y="{sy + FONT_SIZE*0.35:.2f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{FONT_SIZE}" font-family="Arial,sans-serif" font-weight="600" '
            f'fill="{BACKGROUND}" stroke="{BACKGROUND}" stroke-width="3" '
            f'stroke-linejoin="round">{sym}</text>'
        )
        parts.append(
            f'<text x="{sx:.2f}" y="{sy + FONT_SIZE*0.35:.2f}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{FONT_SIZE}" font-family="Arial,sans-serif" font-weight="600" '
            f'fill="{color}">{sym}</text>'
        )

        # H count subscript for non-carbon heteroatoms
        nh = atom.GetTotalNumHs()
        if nh > 0 and sym != 'H':
            hx = sx + FONT_SIZE * 0.5
            hy = sy + FONT_SIZE * 0.35
            h_label = f'H{nh}' if nh > 1 else 'H'
            h_fs = FONT_SIZE * 0.75
            parts.append(
                f'<text x="{hx:.2f}" y="{hy + 2:.2f}" '
                f'text-anchor="start" dominant-baseline="middle" '
                f'font-size="{h_fs:.1f}" font-family="Arial,sans-serif" '
                f'fill="{ATOM_COLORS["H"]}">{h_label}</text>'
            )

    parts.append('</svg>')
    return '\n'.join(parts)
