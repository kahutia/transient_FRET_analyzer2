from sys import platform
# if platform == "darwin":
#     from matplotlib import use
#     use('WXAgg')

import pickle
from tkinter import *
from tkinter import ttk, filedialog
from tkinter import messagebox
from rene_plots import RenePlotSimple
import numpy as np
import os
from pathlib import Path


class ReneMain(Tk):
    def __init__(self):
        super().__init__()
        uip = self.user_parameters(mode='init')

        self.title("Rene main")
        self.geometry("500x470+20+20")
        self.minsize(500, 350)

        self.prepare_gui(uip)
        self.renes = []

    def user_parameters(self, mode='init', uip=()):
        if mode == 'init':
            uip = {
                "time_res": 0.1,
                "gamma": 1.0,
                "leakage": 0.0,
                "bg_d_fix": 0,
                "bg_a_fix": 0,
                "Int_max": 2000,
                "Int_min": -200,
                "toi_start": 0,
                "toi_end": float('Inf'),

                "keyword": "blink",
                "min_peak_len": 0.3,
                "min_peak_int": 0,
                "max_peak_int": float('Inf'),
                "E_tolerance": 0.2,

                "t_bin_size": 50,
                "e_bin_size": 0.01,

                "cmap": 'plasma',

                "bcd_max_n_peaks": 3,
                "bcd_min_occupancy": 0.1,
                "bcd_min_distance": 0.1
            }
        elif mode == 'fetch':
            uip = {
                "time_res": float(self.etr_time_res.get()),
                "gamma": float(self.etr_gamma.get()),
                "leakage": float(self.etr_leakage.get()),
                "bg_d_fix": float(self.etr_bg_d.get()),
                "bg_a_fix": float(self.etr_bg_a.get()),
                "Int_max": float(self.etr_int_max.get()),
                "Int_min": float(self.etr_int_min.get()),
                "toi_start": float(self.etr_toi_start.get()),
                "toi_end": float(self.etr_toi_end.get()),

                "keyword": self.etr_keyword.get(),
                "min_peak_len": float(self.etr_min_peak_len.get()),
                "min_peak_int": float(self.etr_min_peak_int.get()),
                "max_peak_int": float(self.etr_max_peak_int.get()),
                "E_tolerance": float(self.etr_E_tolerance.get()),

                "t_bin_size": int(self.etr_t_bin_size.get()),
                "e_bin_size": float(self.etr_e_bin_size.get()),

                "cmap": self.cmap_selection.get(),

                "bcd_max_n_peaks": int(self.etr_bcd_max_n_peaks.get()),
                "bcd_min_occupancy": float(self.etr_bcd_min_occupancy.get()),
                "bcd_min_distance": float(self.etr_bcd_min_distance.get())
            }
        elif mode == 'set':
            if uip:
                # update uip
                self.update_gui(uip)
                # update GUI fields
        return uip

    def update_gui(self, uip):
        self.etr_min_peak_int.delete(0, END)
        self.etr_min_peak_int.insert(0, str(uip['min_peak_int']))

        self.etr_max_peak_int.delete(0, END)
        self.etr_max_peak_int.insert(0, str(uip['max_peak_int']))

    def prepare_gui(self, uip):
        # --- header ---#
        self.lbl_header = Label(self, text="René Magritte v2.2", width=50, pady=10, font='Helvetica 12 bold').grid(row=0, column=0, columnspan=10)

        # --- column 0, 1 --- #
        self.lbl_time_res = Label(self, text="Time res. (s)", width=10).grid(row=1, column=0)
        self.etr_time_res = Entry(self, textvariable=DoubleVar(self, value=uip["time_res"]), width=7)
        self.etr_time_res.grid(row=1, column=1)

        self.lbl_gamma = Label(self, text="gamma", width=10).grid(row=2, column=0)
        self.etr_gamma = Entry(self, textvariable=DoubleVar(self, value=uip["gamma"]), width=7)
        self.etr_gamma.grid(row=2, column=1)

        self.lbl_leakage = Label(self, text="leakage", width=10).grid(row=3, column=0)
        self.etr_leakage = Entry(self, textvariable=DoubleVar(self, value=uip["leakage"]), width=7)
        self.etr_leakage.grid(row=3, column=1)

        self.lbl_bg_d = Label(self, text="BG. donor", width=10).grid(row=4, column=0)
        self.etr_bg_d = Entry(self, textvariable=DoubleVar(self, value=uip["bg_d_fix"]), width=7)
        self.etr_bg_d.grid(row=4, column=1)

        self.lbl_bg_a = Label(self, text="BG. acceptor", width=10).grid(row=5, column=0)
        self.etr_bg_a = Entry(self, textvariable=DoubleVar(self, value=uip["bg_a_fix"]), width=7)
        self.etr_bg_a.grid(row=5, column=1)

        self.lbl_int_max = Label(self, text="Int. max", width=10).grid(row=7, column=0)
        self.etr_int_max = Entry(self, textvariable=DoubleVar(self, value=uip["Int_max"]), width=7)
        self.etr_int_max.grid(row=7, column=1)

        self.lbl_int_min = Label(self, text="Int. min", width=10).grid(row=8, column=0)
        self.etr_int_min = Entry(self, textvariable=DoubleVar(self, value=uip["Int_min"]), width=7)
        self.etr_int_min.grid(row=8, column=1)

        # Label(self, text=" ", width=10).grid(row=9, column=0) # empty space

        self.lbl_toi_start = Label(self, text="frame start", width=10).grid(row=10, column=0)
        self.etr_toi_start = Entry(self, textvariable=DoubleVar(self, value=uip["toi_start"]), width=7)
        self.etr_toi_start.grid(row=10, column=1)

        self.lbl_toi_end = Label(self, text="frame  end", width=10).grid(row=11, column=0)
        self.etr_toi_end = Entry(self, textvariable=DoubleVar(self, value=uip["toi_end"]), width=7)
        self.etr_toi_end.grid(row=11, column=1)

        cmap_list = ['plasma', 'viridis', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens',
                     'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu',
                     'PuBuGn', 'BuGn', 'YlGn', 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',
                     'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'ocean',
                     'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
                     'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral']
        self.cmap_selection = StringVar(self)
        self.cmap_selection.set(cmap_list[5])
        # self.lbl_cmap = Label(self, text="Color map", width=10).grid(row=12, column=0)
        self.menu_cmap = OptionMenu(self, self.cmap_selection, *cmap_list)
        self.menu_cmap.config(width=9)
        self.menu_cmap.grid(row=12, column=0, columnspan=2)

        # --- column 2, 3 --- #
        self.lbl_keyword = Label(self, text="keyword", width=10).grid(row=1, column=2)
        self.etr_keyword = Entry(self, textvariable=DoubleVar(self, value=uip["keyword"]), width=7)
        self.etr_keyword.grid(row=1, column=3)

        self.lbl_min_peak_len = Label(self, text="min_peak_len", width=10).grid(row=2, column=2)
        self.etr_min_peak_len = Entry(self, textvariable=DoubleVar(self, value=uip["min_peak_len"]), width=7)
        self.etr_min_peak_len.grid(row=2, column=3)

        self.lbl_min_peak_int = Label(self, text="min_peak_int", width=10).grid(row=3, column=2)
        self.etr_min_peak_int = Entry(self, textvariable=DoubleVar(self, value=uip["min_peak_int"]), width=7)
        self.etr_min_peak_int.grid(row=3, column=3)

        self.lbl_max_peak_int = Label(self, text="max_peak_int", width=10).grid(row=4, column=2)
        self.etr_max_peak_int = Entry(self, textvariable=DoubleVar(self, value=uip["max_peak_int"]), width=7)
        self.etr_max_peak_int.grid(row=4, column=3)

        self.lbl_E_tolerance = Label(self, text="E_tolerance", width=10).grid(row=5, column=2)
        self.etr_E_tolerance = Entry(self, textvariable=DoubleVar(self, value=uip["E_tolerance"]), width=7)
        self.etr_E_tolerance.grid(row=5, column=3)

        self.lbl_t_bin_size = Label(self, text="t_bin_size (fr)", width=10).grid(row=7, column=2)
        self.etr_t_bin_size = Entry(self, textvariable=DoubleVar(self, value=uip["t_bin_size"]), width=7)
        self.etr_t_bin_size.grid(row=7, column=3)

        self.lbl_e_bin_size = Label(self, text="e_bin_size", width=10).grid(row=8, column=2)
        self.etr_e_bin_size = Entry(self, textvariable=DoubleVar(self, value=uip["e_bin_size"]), width=7)
        self.etr_e_bin_size.grid(row=8, column=3)

        self.lbl_bcd_max_n_peaks = Label(self, text="# Barcodes", width=10).grid(row=10, column=2)
        self.etr_bcd_max_n_peaks = Entry(self, textvariable=DoubleVar(self, value=uip["bcd_max_n_peaks"]), width=7)
        self.etr_bcd_max_n_peaks.grid(row=10, column=3)

        self.lbl_bcd_min_occupancy = Label(self, text="min_occupancy", width=10).grid(row=11, column=2)
        self.etr_bcd_min_occupancy = Entry(self, textvariable=DoubleVar(self, value=uip["bcd_min_occupancy"]), width=7)
        self.etr_bcd_min_occupancy.grid(row=11, column=3)

        self.lbl_bcd_min_distance = Label(self, text="min_distance", width=10).grid(row=12, column=2)
        self.etr_bcd_min_distance = Entry(self, textvariable=DoubleVar(self, value=uip["bcd_min_distance"]), width=7)
        self.etr_bcd_min_distance.grid(row=12, column=3)

        # --- colum 4, Buttons --- #
        self.btn_get_trace = Button(self, text="get traces", width=10, command=lambda: self.get_trace(uip)).grid(row=1,
                                                                                                                 column=6)
        self.btn_analyse = Button(self, text="re-analyze", width=10, command=self.re_analyze, state=DISABLED).grid(row=3, column=6)
        self.btn_show_ehist = Button(self, text="show hist", width=10, command=self.show_ehist).grid(row=5, column=6)
        self.btn_show_dwell = Button(self, text="show dwell", width=10, command=self.show_dwell).grid(row=6, column=6)
        self.btn_show_trace = Button(self, text="show trace", width=10, command=self.show_trace).grid(row=7, column=6)
        self.btn_save_result = Button(self, text="save result", width=10, command=self.save_result).grid(row=9, column=6)
        self.btn_load_result = Button(self, text="load rene", width=10, command=self.load_rene).grid(row=10, column=6)

        # --- log and progress bar --- #
        self.log_text = StringVar(self, 'No data loaded.')
        self.lbl_logtxt = Label(self, textvariable=self.log_text, height=4, wraplength=480, justify=LEFT, padx=10)
        self.lbl_logtxt.grid(row=20, column=0, columnspan=10)

        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=500, mode='determinate')
        self.progress_bar.grid(column=0, row=21, columnspan=10)
        self.progress_bar["maximum"] = 100
        self.set_progressbar(0)

    def get_pg_data(self, mode='get', init_dir=''):
        py_path = os.getcwd()
        pgdata_path = os.path.join(py_path, 'renemain.pgdat')
        if mode == 'get':
            # --- get preivious usage record
            if os.path.isfile(pgdata_path):
                with open(pgdata_path, 'r') as pgdata_file:
                    init_dir = pgdata_file.read()
            else:
                init_dir = "/Users/sunghyunkim/Desktop/"
        else:   # save
            with open(pgdata_path, 'w') as pgdata_file:
                pgdata_file.write(f"{init_dir}")
        return init_dir

    def get_trace(self, uip):
        uip = self.user_parameters('fetch')

        # --- get previous usage record
        init_dir = self.get_pg_data(mode='get')
        # --- get trace file path from user
        trace_files = filedialog.askopenfiles(mode="r", initialdir=init_dir, title="Select file",
                                             filetypes=[('trace files', ['.traces', '.traces2', '.dat', '.txt'])])
        # --- save usage record
        self.get_pg_data(mode='set', init_dir=os.path.dirname(trace_files[0].name))

        # --- run rene
        c_pth = list()
        for trf in trace_files:
            c_pth.append(trf.name)

        self.set_progressbar(20)
        self.renes = []
        # with Rene main, only one rene is allowed to be loaded
        self.renes.append(RenePlotSimple(c_pth, uip, self.progress_bar, self.log_text, gui_obj=self))

        self.show_ehist()
        # self.show_dwell()
        self.save_result()

    def re_analyze(self):
        uip = self.user_parameters('fetch')
        if len(self.renes) == 0:
            messagebox.showinfo(title='Rene', message='Load data first!')
        for rene in self.renes:
            rene.set_parameters(uip=uip)
            rene.run_analysis()
        self.show_ehist()
        # self.show_dwell()

    def show_dwell(self):
        self.log_text.set('preparing figures (this may take a few minutes) ...\n')
        self.progress_bar["value"] = 5
        self.progress_bar.update()

        uip = self.user_parameters('fetch')
        for rene in self.renes:
            rene.set_parameters(uip=uip)

        n_rene = len(self.renes)
        for rene_id, rene in enumerate(self.renes):
            fhd_dwell = rene.show_dwell(700, rene_id)
            output_pth = os.path.dirname(rene.tr_path[0])
            f_name_base, f_name_ext = os.path.splitext(os.path.basename(rene.tr_path[0]))
            if 'trace' in f_name_ext:
                file_keywords = f_name_base
            else:
                file_keywords = ''
            fhd_dwell.savefig(os.path.join(output_pth, 'dwells_' + file_keywords + '.eps'), format='eps')

        self.log_text.set('Job done!!!\n')
        self.progress_bar["value"] = 100
        self.progress_bar.update()

    def show_ehist(self):
        self.log_text.set('preparing figures (this may take a few minutes) ...\n')
        self.progress_bar["value"] = 5
        self.progress_bar.update()

        uip = self.user_parameters('fetch')
        for rene in self.renes:
            rene.set_parameters(uip=uip)

        # --- draw hist-kymo --- #
        self.fhd_hist_kymo = []
        for rene_id, rene in enumerate(self.renes):
            # --- prepare file path for figure saving
            output_pth = os.path.dirname(rene.tr_path[0])
            f_name_base, f_name_ext = os.path.splitext(os.path.basename(rene.tr_path[0]))
            if 'trace' in f_name_ext:
                file_keywords = f_name_base
            else:
                file_keywords = ''
            output_pth = os.path.join(output_pth, 'hist_kymo_' + file_keywords + '.eps')

            # --- draw figure
            fhd, self.fit_btnhd = rene.show_hist_kymo(fignum=600, rene_id=rene_id, output_file_path=output_pth)
            # fhd.savefig(output_pth, format='eps') # this code is move to rene_plot

            self.fhd_hist_kymo.append(fhd)

        self.log_text.set('Job done!!!\n')
        self.progress_bar["value"] = 100
        self.progress_bar.update()

    def show_trace(self):
        self.log_text.set('preparing figures...\n')
        self.progress_bar["value"] = 5
        self.progress_bar.update()

        uip = self.user_parameters('fetch')
        for rene in self.renes:
            rene.set_parameters(uip=uip)

        for rene_id, rene in enumerate(self.renes):
            # --- prepare file path for figure saving
            output_pth = os.path.dirname(rene.tr_path[0])
            f_name_base, f_name_ext = os.path.splitext(os.path.basename(rene.tr_path[0]))
            if 'trace' in f_name_ext:
                file_keywords = f_name_base
            else:
                file_keywords = ''
            output_pth = os.path.join(output_pth, 'hist_kymo_' + file_keywords + '.eps')

            fhd_dwell = rene.show_trace(900, rene_id, output_pth)
            # output_pth = os.path.dirname(rene.tr_path[0])
            # f_name_base, f_name_ext = os.path.splitext(os.path.basename(rene.tr_path[0]))
            # if 'trace' in f_name_ext:
            #     file_keywords = f_name_base
            # else:
            #     file_keywords = ''
            # fhd_dwell.savefig(os.path.join(output_pth, 'dwells_' + file_keywords + '.eps'), format='eps')

        self.log_text.set('Job done!!!\n')
        self.progress_bar["value"] = 100
        self.progress_bar.update()

    def save_result(self):
        for rene_id, rene in enumerate(self.renes):
            file_name_base = os.path.basename(rene.tr_path[0]).replace('.traces', '')
            output_dir = os.path.dirname(rene.tr_path[0])

            # --- save rene

            # remove tkinter handle that belongs to the main GUI
            del rene.pg_bar
            del rene.log_txt
            del rene.gui_obj
            with open(os.path.join(output_dir, file_name_base + '.rene'), 'wb') as rene_file:
                pickle.dump(rene, rene_file)
                rene_file.close()
            # restore tkinter handles
            rene.log_txt = self.log_text
            rene.pg_bar = self.progress_bar
            rene.gui_obj = self

            # --- save dwell data
            tmp_out = []
            for mid in range(rene.N_trace):
                tmp0 = rene.E_per_peak_sel[mid]
                tmp1 = rene.I_per_peak_sel[mid]
                # tmp2 = [len(elmts) * rene.time_res for elmts in rene.I_in_peak_sel[mid]]
                tmp2 = [len(elmts) * rene.uip['time_res'] for elmts in rene.I_in_peak_sel[mid]]
                for i, j, k in zip(tmp0, tmp1, tmp2):
                    tmp_out.append([i, j, k])
            np.savetxt(os.path.join(output_dir, file_name_base + '_binding_events_EIT.dat'), tmp_out)

    def load_rene(self):
        self.log_text.set('loading rene...\n')
        self.progress_bar["value"] = 5
        self.progress_bar.update()

        # --- get preivious usage record
        init_dir = self.get_pg_data(mode='get')
        # --- get rene file path from user
        trace_files = filedialog.askopenfiles(mode="r", initialdir=init_dir, title="Select file",
                                              filetypes=[('rene files', ['.rene'])])
        # --- save usage record
        self.get_pg_data(mode='set', init_dir=os.path.dirname(trace_files[0].name))

        self.renes = []
        self.renes.append(pickle.load(open(trace_files[0].name, "rb")))

        self.renes[0].log_txt = self.log_text
        self.renes[0].pg_bar = self.progress_bar

        # self.user_parameters('set') # update rene parameter to the main GUI

        self.log_text.set('rene file loaded!\n')
        self.progress_bar["value"] = 100
        self.progress_bar.update()

    def set_progressbar(self, c_progress):
        self.progress_bar["value"] = c_progress
        self.progress_bar.update()


renemain = ReneMain()
renemain.mainloop()
