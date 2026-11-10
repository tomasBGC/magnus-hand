#!/usr/bin/env python3
"""
GUI launcher for the eFlesh microstructure pipeline.
Double-click run_eflesh.bat to open this window.
"""

import io, json, math, os, subprocess, sys, tempfile, threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference grid values (same as cut-cell.ipynb)
GRID_C = [1, 2, 3, 4, 5, 6, 7, 8, 16]
GRID_E = [0.05, 0.1, 0.2, 0.3, 0.5]
PRINT_THRESHOLD = 0.4  # mm — FDM 0.4 mm nozzle


# ---------------------------------------------------------------------------
# Material2Geometry helpers
# ---------------------------------------------------------------------------

def _load_m2g():
    matopt = os.path.join(os.path.dirname(SCRIPT_DIR), 'matopt')
    path   = os.path.join(matopt, 'tools/material2geometry')
    if not os.path.isdir(path):
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from material2geometry import Material2Geometry
        coeffs = os.path.join(path, '0646_geo_1_coeffs.txt')
        return Material2Geometry(in_path=coeffs)
    except Exception:
        return None


def min_feature(m2g, nu, E, c):
    """Minimum geometry dimension in mm, matching cut-cell.ipynb: params[5:-4] * c."""
    import numpy as np
    params = m2g.evaluate(nu, E)
    return float(np.min(np.array(params[5:-4]) * c))


# ---------------------------------------------------------------------------
# Pouch dialog
# ---------------------------------------------------------------------------

class PouchDialog(tk.Toplevel):
    FIELDS = [
        ('Diameter (mm)',    'diameter',  '4.76'),
        ('Thickness (mm)',   'thickness', '1.59'),
        ('Position X (mm)', 'x',         '5.0'),
        ('Position Y (mm)', 'y',         '0.0'),
        ('Position Z (mm)', 'z',         '0.0'),
        ('Rotation X (°)',  'rx',        '0.0'),
        ('Rotation Y (°)',  'ry',        '90.0'),
        ('Rotation Z (°)',  'rz',        '0.0'),
    ]

    def __init__(self, parent, title='Add Pouch', values=None):
        super().__init__(parent)
        self.title(title)
        self.resizable(False, False)
        self.result = None
        self._ents  = {}

        for row, (label, key, default) in enumerate(self.FIELDS):
            ttk.Label(self, text=label).grid(row=row, column=0, padx=10, pady=3, sticky='e')
            e = ttk.Entry(self, width=12)
            e.insert(0, str((values or {}).get(key, default)))
            e.grid(row=row, column=1, padx=10, pady=3)
            self._ents[key] = e

        bf = ttk.Frame(self)
        bf.grid(row=len(self.FIELDS), column=0, columnspan=2, pady=8)
        ttk.Button(bf, text='OK',     command=self._ok).pack(side='left', padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side='left', padx=4)

        self.transient(parent)
        self.grab_set()
        self.wait_window()

    def _ok(self):
        try:
            self.result = {k: float(e.get()) for k, e in self._ents.items()}
        except ValueError:
            messagebox.showerror('Input error', 'All fields must be numeric.', parent=self)
            return
        self.destroy()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('eFlesh Microstructure Generator')
        self.minsize(760, 640)

        self._stl     = tk.StringVar()
        self._E       = tk.StringVar(value='0.05')
        self._nu      = tk.StringVar(value='0.3')
        self._cell    = tk.StringVar(value='6')
        self._blender = tk.StringVar(value=self._find_blender())
        self._pouches = [
            {'diameter': 4.76, 'thickness': 1.59, 'x': 5, 'y': 0, 'z': -5, 'rx': 0, 'ry': 90, 'rz': 0},
            {'diameter': 4.76, 'thickness': 1.59, 'x': 5, 'y': 0, 'z':  5, 'rx': 0, 'ry': 90, 'rz': 0},
        ]
        self._m2g    = None
        self._sel_lbl = None

        self._build_ui()
        self._refresh_pouches()

    @staticmethod
    def _find_blender():
        for ver in ['5.1', '5.0', '4.4', '4.3', '4.2', '4.1', '4.0']:
            p = rf'C:\Program Files\Blender Foundation\Blender {ver}\blender.exe'
            if os.path.exists(p):
                return p
        return r'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill='both', expand=True, padx=6, pady=6)
        self._build_setup()
        self._build_printability()
        self._build_log()

    def _build_setup(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text='  Setup  ')

        # STL picker
        lf = ttk.LabelFrame(f, text='Input STL')
        lf.pack(fill='x', padx=8, pady=5)
        ttk.Entry(lf, textvariable=self._stl, width=62).pack(
            side='left', padx=4, pady=4, fill='x', expand=True)
        ttk.Button(lf, text='Browse…', command=self._browse_stl).pack(side='left', padx=4)

        # Mesh parameters
        lf = ttk.LabelFrame(f, text='Mesh Parameters')
        lf.pack(fill='x', padx=8, pady=5)
        for label, var, unit in [
            ("Young's modulus E", self._E,    'MPa'),
            ("Poisson's ratio ν", self._nu,   ''),
            ('Cell size',         self._cell, 'mm'),
        ]:
            sub = ttk.Frame(lf)
            sub.pack(side='left', padx=18, pady=4)
            ttk.Label(sub, text=label).pack()
            ttk.Entry(sub, textvariable=var, width=9).pack()
            ttk.Label(sub, text=unit, foreground='gray').pack()

        # Blender path
        lf = ttk.LabelFrame(f, text='Blender Executable')
        lf.pack(fill='x', padx=8, pady=5)
        ttk.Entry(lf, textvariable=self._blender, width=62).pack(
            side='left', padx=4, pady=4, fill='x', expand=True)
        ttk.Button(lf, text='Browse…', command=self._browse_blender).pack(side='left', padx=4)

        # Pouches table
        lf = ttk.LabelFrame(f, text='Magnet Pouches')
        lf.pack(fill='both', expand=True, padx=8, pady=5)
        cols   = ('#', 'Ø mm', 'Thick mm', 'X', 'Y', 'Z', 'Rot X°', 'Rot Y°', 'Rot Z°')
        widths = [30, 65, 78, 55, 55, 55, 62, 62, 62]
        self._ptree = ttk.Treeview(lf, columns=cols, show='headings', height=5)
        for c, w in zip(cols, widths):
            self._ptree.heading(c, text=c)
            self._ptree.column(c, width=w, anchor='center')
        self._ptree.pack(fill='both', expand=True, padx=4, pady=4)
        self._ptree.bind('<Double-1>', lambda _: self._edit_pouch())

        bf = ttk.Frame(lf)
        bf.pack(pady=4)
        ttk.Button(bf, text='Add',    command=self._add_pouch).pack(side='left', padx=4)
        ttk.Button(bf, text='Edit',   command=self._edit_pouch).pack(side='left', padx=4)
        ttk.Button(bf, text='Remove', command=self._remove_pouch).pack(side='left', padx=4)

        # Action row
        af = ttk.Frame(f)
        af.pack(fill='x', padx=8, pady=8)
        ttk.Button(af, text='Preview Printability', command=self._preview_print).pack(side='left', padx=4)
        ttk.Button(af, text='Run Pipeline',         command=self._run).pack(side='right', padx=4)

    def _build_printability(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text='  Printability  ')

        ttk.Label(
            f,
            text='Minimum feature size (mm) — same method as cut-cell.ipynb   '
                 '(threshold: 0.4 mm, FDM 0.4 mm nozzle)',
            foreground='gray',
        ).pack(anchor='w', padx=8, pady=(6, 2))

        e_cols = [f'E={e}' for e in GRID_E]
        cols   = ['Cell size (c)'] + e_cols
        self._print_tree = ttk.Treeview(f, columns=cols, show='headings')
        self._print_tree.heading('Cell size (c)', text='Cell size (c)')
        self._print_tree.column('Cell size (c)', width=100, anchor='center')
        for label in e_cols:
            self._print_tree.heading(label, text=label)
            self._print_tree.column(label, width=105, anchor='center')
        self._print_tree.pack(fill='both', expand=True, padx=8, pady=4)

        self._print_tree.tag_configure('selected', background='#ffffcc')
        self._print_tree.tag_configure('fail',     foreground='#cc0000')
        self._print_tree.tag_configure('pass',     foreground='#006400')

        self._sel_lbl = ttk.Label(f, text='', font=('TkDefaultFont', 10, 'bold'))
        self._sel_lbl.pack(anchor='w', padx=12, pady=(4, 8))

    def _build_log(self):
        f = ttk.Frame(self._nb)
        self._nb.add(f, text='  Log  ')
        self._log = scrolledtext.ScrolledText(f, state='disabled', font=('Consolas', 9))
        self._log.pack(fill='both', expand=True, padx=8, pady=(8, 2))
        ttk.Button(f, text='Clear log', command=self._clear_log).pack(
            anchor='e', padx=8, pady=4)

    # ---------------------------------------------------------- Pouch mgmt --

    def _refresh_pouches(self):
        for item in self._ptree.get_children():
            self._ptree.delete(item)
        for i, p in enumerate(self._pouches, 1):
            self._ptree.insert('', 'end', values=(
                i,
                f"{p['diameter']:.2f}", f"{p['thickness']:.2f}",
                f"{p['x']:.1f}",  f"{p['y']:.1f}",  f"{p['z']:.1f}",
                f"{p['rx']:.1f}", f"{p['ry']:.1f}", f"{p['rz']:.1f}",
            ))

    def _add_pouch(self):
        dlg = PouchDialog(self, 'Add Pouch')
        if dlg.result:
            self._pouches.append(dlg.result)
            self._refresh_pouches()

    def _edit_pouch(self):
        sel = self._ptree.selection()
        if not sel:
            return
        idx = self._ptree.index(sel[0])
        dlg = PouchDialog(self, 'Edit Pouch', self._pouches[idx])
        if dlg.result:
            self._pouches[idx] = dlg.result
            self._refresh_pouches()

    def _remove_pouch(self):
        sel = self._ptree.selection()
        if not sel:
            return
        self._pouches.pop(self._ptree.index(sel[0]))
        self._refresh_pouches()

    # ----------------------------------------------------------- File pickers

    def _browse_stl(self):
        p = filedialog.askopenfilename(
            title='Select STL file',
            filetypes=[('STL files', '*.stl'), ('All files', '*.*')],
        )
        if p:
            self._stl.set(p)

    def _browse_blender(self):
        p = filedialog.askopenfilename(
            title='Select blender.exe',
            filetypes=[('Executable', '*.exe'), ('All files', '*.*')],
        )
        if p:
            self._blender.set(p)

    # ------------------------------------------------------- Printability ---

    def _get_params(self):
        try:
            return float(self._E.get()), float(self._nu.get()), float(self._cell.get())
        except ValueError:
            messagebox.showerror('Error', 'E, ν, and cell size must be numbers.')
            return None

    def _preview_print(self):
        p = self._get_params()
        if p is None:
            return
        E, nu, cell = p
        self._fill_print_table(nu, E, cell)
        self._nb.select(1)

    def _fill_print_table(self, nu, E_sel, c_sel):
        for item in self._print_tree.get_children():
            self._print_tree.delete(item)
        if self._sel_lbl:
            self._sel_lbl.config(text='')

        if self._m2g is None:
            self._m2g = _load_m2g()
        if self._m2g is None:
            self._print_tree.insert('', 'end',
                values=['material2geometry not available'] + [''] * len(GRID_E))
            return

        for c in GRID_C:
            row = [str(c)]
            all_fail = True
            for E in GRID_E:
                d = min_feature(self._m2g, nu, E, c)
                if d < PRINT_THRESHOLD:
                    row.append('Unprintable')
                else:
                    row.append(f'{d:.4f}')
                    all_fail = False

            is_sel = abs(c - c_sel) < 0.5
            if is_sel:
                tag = 'selected'
            elif all_fail:
                tag = 'fail'
            else:
                tag = 'pass'
            self._print_tree.insert('', 'end', values=row, tags=(tag,))

        # Summary for exact selection
        try:
            d_sel = min_feature(self._m2g, nu, E_sel, c_sel)
            ok    = d_sel >= PRINT_THRESHOLD
            mark  = '✓  Printable' if ok else '✗  Unprintable  (below 0.4 mm threshold)'
            color = '#006400' if ok else '#cc0000'
            self._sel_lbl.config(
                text=f"Your settings: E={E_sel}, ν={nu}, c={c_sel}"
                     f"  →  min feature = {d_sel:.4f} mm    {mark}",
                foreground=color,
            )
        except Exception:
            pass

    # ------------------------------------------------------------ Pipeline --

    def _run(self):
        p = self._get_params()
        if p is None:
            return
        E, nu, cell = p
        stl = self._stl.get()
        if not os.path.exists(stl):
            messagebox.showerror('Error', 'Please select a valid STL file.')
            return
        self._nb.select(2)
        threading.Thread(
            target=self._pipeline_thread, args=(stl, E, nu, cell), daemon=True
        ).start()

    def _log_write(self, text):
        def _do():
            self._log.config(state='normal')
            self._log.insert('end', text)
            self._log.see('end')
            self._log.config(state='disabled')
        self.after(0, _do)

    def _clear_log(self):
        self._log.config(state='normal')
        self._log.delete('1.0', 'end')
        self._log.config(state='disabled')

    def _stream(self, cmd):
        """Run cmd, stream stdout+stderr to the log widget line by line."""
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, encoding='utf-8', errors='replace',
        )
        for line in proc.stdout:
            self._log_write(line)
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'Process exited with code {proc.returncode}')

    def _pipeline_thread(self, stl, E, nu, cell):
        try:
            import generate_eflesh as pl
        except ImportError as exc:
            self._log_write(f'[ERROR] Cannot import generate_eflesh: {exc}\n')
            return

        matopt      = os.path.join(os.path.dirname(SCRIPT_DIR), 'matopt')
        cut_cli     = os.path.join(SCRIPT_DIR, 'build', 'isosurface_inflator', 'cut_cells_cli')
        blender_exe = self._blender.get()
        magnets     = [
            [p['diameter'], p['thickness'],
             [p['x'], p['y'], p['z']],
             [p['rx'], p['ry'], p['rz']]]
            for p in self._pouches
        ]

        # 1 — data.json
        self._log_write('[1/4] Generating data.json ...\n')
        buf = io.StringIO()
        _out, sys.stdout = sys.stdout, buf
        try:
            input_obj, out_path, data_json = pl.generate_data_json(
                stl, cell, E, nu, SCRIPT_DIR, matopt)
        except Exception as exc:
            sys.stdout = _out
            self._log_write(f'[ERROR] {exc}\n'); return
        sys.stdout = _out
        self._log_write(buf.getvalue())

        # 2 — fix WSL paths
        self._log_write('[2/4] Fixing paths for WSL ...\n')
        try:
            pl.fix_paths_in_json(data_json)
        except Exception as exc:
            self._log_write(f'[ERROR] {exc}\n'); return

        # 3 — cut_cells_cli
        self._log_write('[3/4] Running cut_cells_cli via WSL ...\n')
        cmd = [
            'wsl', pl.win_to_wsl(cut_cli),
            '-p', pl.abs_wsl(data_json),
            '--surface', pl.abs_wsl(input_obj),
            '--gridSize', str(cell),
            '-o', pl.abs_wsl(out_path),
            '-r', '50',
        ]
        try:
            self._stream(cmd)
        except Exception as exc:
            self._log_write(f'[ERROR] {exc}\n'); return

        # 4 — Blender
        final = os.path.splitext(out_path)[0] + '_with_pouches.obj'
        self._log_write('[4/4] Running Blender to add pouches ...\n')
        try:
            self._run_blender(blender_exe, out_path, final, magnets)
        except Exception as exc:
            self._log_write(f'[ERROR] {exc}\n'); return

        self._log_write(f'\nDone.  Output: {final}\n')
        self.after(0, lambda: self._fill_print_table(nu, E, cell))

    def _run_blender(self, blender_exe, input_obj, output_obj, magnets):
        driver = f"""\
import sys, math
sys.path.insert(0, {repr(SCRIPT_DIR)})
import create_pouch, bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
bpy.ops.wm.obj_import(filepath={repr(os.path.abspath(input_obj))})
bpy.context.view_layer.objects.active.name = "micro"

for idx, mag in enumerate({json.dumps(magnets)}):
    rot = mag[3] if len(mag) > 3 else [0, 90, 0]
    ori = tuple(math.radians(d) for d in rot)
    hull, pouch = create_pouch.create_pouch(
        mag[0], mag[1], mag[2], "pouch" + str(idx), orientation=ori)
    create_pouch.boolean("micro", hull,  "DIFFERENCE")
    create_pouch.boolean("micro", pouch, "UNION")

create_pouch.export_mesh({repr(os.path.abspath(output_obj))})
print("Exported:", {repr(os.path.abspath(output_obj))})
"""
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8') as t:
            t.write(driver)
            tmp = t.name
        try:
            self._stream([blender_exe, '--background', '--python', tmp])
        finally:
            os.unlink(tmp)


if __name__ == '__main__':
    App().mainloop()
