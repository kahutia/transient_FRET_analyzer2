from random import random
import numpy as np
import os
from pathlib import Path
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import optimize
from rene import Rene
from matplotlib.widgets import Slider, Button, TextBox
from sklearn.mixture import GaussianMixture


def gen_color_code(nx, ny):
    # generate color code
    color_code_dark = []
    color_code_bright = []
    for mid in range(nx):
        tmp_dark = []
        tmp_bright = []
        for pid in range(ny):
            tc_code = [random() * 0.8 + 0.15, random() * 0.8, random() * 0.6]
            tmp_dark.append([elmt * 0.8 for elmt in tc_code])
            tmp_bright.append([elmt / 3 + 0.6 for elmt in tc_code])
        color_code_dark.append(tmp_dark)
        color_code_bright.append(tmp_bright)
    return color_code_dark, color_code_bright


class RenePlotSimple(Rene):

    def update_barcodes(self, bcd_ax, mid=-1):
        bcd_ax.cla()
        if mid == -1:
            c_bar_pos = self.barcode_pos_peak
        else:
            c_bar_pos = self.barcode_pos_peak_idv[mid]

        if not hasattr(c_bar_pos, "__len__"):
            c_bar_pos = [c_bar_pos]
        for c_cen in c_bar_pos:
            dummy_x = [c_cen, c_cen]
            bcd_ax.plot(dummy_x, [0, 1], color=(0, 0, 0), lw=3)
            bcd_ax.set_xlabel('FRET')
            bcd_ax.set_ylabel('Barcode')
            bcd_ax.set_ylim(0, 1)
            bcd_ax.set_xlim(self.e_bin[0], self.e_bin[-1])
            bcd_ax.tick_params(axis='x', which='both', bottom=True, top=True, labeltop=False, labelbottom=True)
            bcd_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
        plt.show(block=False)
        plt.pause(0.001)

    def update_histogram(self, hist_ax, mid=-1):
        if mid == -1:
            hist_data = self.peak_hist
        else:
            hist_data = self.peak_hist_idv[mid]

        hist_ax.cla()
        e_bin_center = np.add(self.e_bin[0:-1], (self.e_bin[2] - self.e_bin[1]) / 2)
        hist_ax.bar(e_bin_center, hist_data, self.e_bin[1] - self.e_bin[0], color=(0.7, 0.7, 0.7))
        hist_ax.set_xlim(self.e_bin[0], self.e_bin[-1])
        hist_ax.set_ylabel('# peaks')
        hist_ax.tick_params(axis='x', which='both', bottom=True, top=True, labeltop=False, labelbottom=False)

    def show_hist_kymo(self, fignum, rene_id=0, mid=-1, output_file_path=[]):
        n_line_kimo, n_e_bin = np.shape(self.hist_kymo)
        t_grid, e_grid = np.mgrid[0:n_line_kimo, self.e_bin[0]:self.e_bin[-1]:(self.e_bin[1] - self.e_bin[0])]
        t_grid *= self.time_res * self.t_bin_size
        e_bin_center = np.add(self.e_bin[0:-1], (self.e_bin[2] - self.e_bin[1]) / 2)

        color_code_dark, color_code_bright = gen_color_code(nx=self.N_trace, ny=int(self.trace_len / 3))

        # --- prepare figure for kymograph
        fhd_kymograph_peak = plt.figure(fignum + 1, figsize=[4.5, 8])
        fhd_kymograph_peak.clf()
        ax_kymos = []

        # --- draw hist kymograph as number of peaks
        gs = fhd_kymograph_peak.add_gridspec(6, 1)
        ax_kymos.append(fhd_kymograph_peak.add_subplot(gs[0:-2, 0]))

        # draw individual binding evetns
        mid_c = -1
        for tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes in zip(self.E_in_peak_sel, self.trans_d2u_sel,
                                                                       color_code_bright):
            mid_c += 1
            if mid == -1 or mid == mid_c:
                for tmp_etrain, tmp_t_start, c_code in zip(tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes):
                    tmp_fr_vct = [(tmp_t_start + elmt) * self.time_res for elmt in range(len(tmp_etrain))]
                    plt.plot(tmp_etrain, tmp_fr_vct, color=c_code, linewidth=0.5)
        mid_c = -1
        for tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes in zip(self.E_per_peak_sel, self.trans_d2u_sel,
                                                                       color_code_dark):
            mid_c += 1
            if mid == -1 or mid == mid_c:
                for tmp_e_value, tmp_t_start, c_code in zip(tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes):
                    plt.plot(tmp_e_value, tmp_t_start * self.time_res, '.', color=c_code, markersize=2)

        plt.ylabel('Time (s)')
        plt.xlim(self.e_bin[0], self.e_bin[-1])
        plt.ylim(t_grid[-1][0], 0)
        plt.tick_params(axis='x', which='both', bottom=False, top=True, labeltop=True, labelbottom=False)

        # --- draw FRET histogram
        ax_kymos.append(fhd_kymograph_peak.add_subplot(gs[-2, 0]))
        self.update_histogram(ax_kymos[1], mid)

        # --- add fit button
        ax_fit = plt.axes([0.93, 0.3, 0.05, 0.075])
        btn_fit = Button(ax_fit, 'Fit')
        btn_fit.on_clicked(
            lambda event: self.fig_ehist(event, e_bin_center, ax_kymos, method='curv_fit', automode='manual', mid=mid, output_file_path=output_file_path, fhd_kymograph_peak=fhd_kymograph_peak))

        # --- Draw fit curve and barcodes
        ax_kymos.append(fhd_kymograph_peak.add_subplot(gs[-1, 0]))
        self.fig_ehist('', e_bin_center, ax_kymos, method='curv_fit', automode='auto', mid=mid)
        plt.show(block=False)
        plt.pause(0.001)

        # --- Save kymo
        self.save_kymograph_fig(output_file_path, fhd_kymograph_peak, mid)
        self.save_barcode_info(output_file_path)

        return fhd_kymograph_peak, btn_fit

    def save_kymograph_fig(self, output_file_path, fhd, mid):
        if output_file_path:
            # --- save kymo
            fhd.savefig(output_file_path, format='eps')

    def save_barcode_info(self, output_file_path):
            # --- save barcode positions in ascii format
            bcd_out_dir = os.path.dirname(output_file_path)
            bcd_fname = os.path.basename(output_file_path).replace('hist_kymo', 'bcd_info').replace('.eps', '.txt')
            with open(os.path.join(bcd_out_dir, bcd_fname), "w") as bcd_file:
                bcd_file.write(f'<All molecules>\n')
                for bcd_id in range(len(self.barcode_pos_peak)):
                    bcd_file.write(f'Barcode {bcd_id}:\n')
                    bcd_file.write(f' position {self.barcode_pos_peak[bcd_id]}\n')
                    bcd_file.write(f' width {self.barcode_width_peak[bcd_id]}\n')
                    bcd_file.write(f' weight {self.barcode_weight_peak[bcd_id]}\n')
                for tmp_mid in range(self.N_trace):
                    bcd_file.write(f'\n<Molecules {tmp_mid}>\n')
                    if self.barcode_pos_peak_idv[tmp_mid]:
                        for bcd_id in range(len(self.barcode_pos_peak_idv[tmp_mid])):
                            bcd_file.write(f'Barcode {bcd_id}:\n')
                            bcd_file.write(f' position {self.barcode_pos_peak_idv[tmp_mid][bcd_id]}\n')
                            bcd_file.write(f' width {self.barcode_width_peak_idv[tmp_mid][bcd_id]}\n')
                            bcd_file.write(f' weight {self.barcode_weight_peak_idv[tmp_mid][bcd_id]}\n')
                    else:
                        bcd_file.write('No barcode found\n')

    def fig_ehist(self, event, x_data, ax, method='curv_fit', automode='manual', mid=-1, output_file_path=[], fhd_kymograph_peak=[]):
        self.update_log(f"\r   Cleaning up the figure ({method})... ")
        self.set_progressbar(5)

        # remove previous fit curves
        self.update_histogram(ax[1], mid)

        # start curve fit
        def fit_gauss(x_data, y_data, fc_x, inits_pos):
            def fn_gauss_multi(x, *p):  # a, xc, w = p
                _n_peaks = int(len(p) / 3)
                y = 0.0
                for pid in range(_n_peaks):
                    y += np.exp(-(x - p[pid * 3 + 1]) ** 2 / p[pid * 3 + 2] ** 2 / 2) * p[pid * 3] / p[pid * 3 + 2] / np.sqrt(2 * np.pi)
                return y
            n_peaks = len(inits_pos)
            para_init = []
            para_bounds = []
            for pki in range(n_peaks):
                para_init += [max(y_data)*(x_data[1]-x_data[0]), inits_pos[pki], 0.02]
                para_bounds += [np.Inf, 1.1, 0.05]
            # _fc_init = fn_gauss_multi(fc_x, *para_init)
            para_bounds = ([0.0, 0.0, 0.0] * n_peaks, para_bounds)
            coeff, var_matrix = optimize.curve_fit(fn_gauss_multi, x_data, y_data, p0=para_init, bounds=para_bounds, maxfev=1000)
            _fc = fn_gauss_multi(fc_x, *coeff)
            _barcodes_pos = []
            _barcodes_weight = []
            _barcodes_width = []
            for pki in range(n_peaks):
                _barcodes_weight.append(coeff[pki * 3 + 0])
                _barcodes_pos.append(coeff[pki * 3 + 1])
                _barcodes_width.append(coeff[pki * 3 + 2])

            norm_factor = np.sum(_barcodes_weight)
            _barcodes_weight = [elmt/norm_factor for elmt in _barcodes_weight]

            return _fc, _barcodes_pos, _barcodes_weight, _barcodes_width

        if automode == 'auto':
            self.update_log(f"\r   Auto-detection started ({method})... ")
            self.set_progressbar(10)
            if mid == -1:
                coords = self.barcode_pos_peak
            else:
                coords = self.barcode_pos_peak_idv[mid]
        else:  # get peak position from the user
            self.update_log(f"\r   waiting for user input ({method})... ")
            self.set_progressbar(10)

            coords = plt.ginput(99, timeout=0)
            coords = [coord[0] for coord in coords]

        fc_x_vct = np.arange(x_data[0], x_data[-1], (x_data[1] - x_data[0]) / 10)
        if mid == -1:
            y_data = self.peak_hist
        else:
            y_data = self.peak_hist_idv[mid]
        barcode_found = True
        try:
            fc, barcodes_pos, barcodes_weight, barcodes_width = fit_gauss(x_data=x_data, y_data=y_data, fc_x=fc_x_vct, inits_pos=coords)
        except:
            print('no barcode found')
            barcode_found = False
            barcodes_pos = []
            barcodes_weight = []
            barcodes_width = []
            fc = []

        self.update_log(f"\r   Updating data ({method})... ")
        self.set_progressbar(50)

        # update rene data
        if mid == -1:
            self.barcode_pos_peak = barcodes_pos
            self.barcode_width_peak = barcodes_width
            self.barcode_weight_peak = barcodes_weight
        else:
            self.barcode_pos_peak_idv[mid] = barcodes_pos
            self.barcode_width_peak_idv[mid] = barcodes_width
            self.barcode_weight_peak_idv[mid] = barcodes_weight

        # update figures
        self.update_log(f"\r   Updating figures ({method})... ")
        self.set_progressbar(70)

        if automode == 'manual':
            self.update_histogram(ax[1], mid)

        if barcode_found:
            # --- draw fit curv
            plt.sca(ax[1])
            plt.plot(fc_x_vct, fc, 'r')
            plt.ylim(0, max(fc)*1.3)
            for bcd_id in range(len(barcodes_pos)):
                tmp_y_pointer = (np.abs(fc_x_vct - barcodes_pos[bcd_id])).argmin()
                plt.text(barcodes_pos[bcd_id]-0.1, fc[tmp_y_pointer] * 1.1,
                         f'{barcodes_pos[bcd_id]:.3f}$\pm${barcodes_width[bcd_id]:.3f}', fontsize=7)
            # --- draw barcode
            self.update_barcodes(ax[2], mid)
        else:
            ax[2].cla()

        self.update_log(f"\r   Saving figures ({method})... ")
        self.set_progressbar(85)

        print(output_file_path)
        if output_file_path:
            self.save_barcode_info(output_file_path)
            if mid == -1:
                self.save_kymograph_fig(output_file_path, fhd_kymograph_peak, mid)


        self.update_log(f"\r   Job done! ({method})... ")
        self.set_progressbar(100)

    def show_dwell(self, fignum, rene_id):
        # --- get FRET range
        def get_thresh(min_val, max_val, minormax='min'):
            user_response = True
            while user_response:
                plt.figure(fignum)
                plt.clf()
                for mid in range(self.N_trace):
                    plt.plot(self.E_per_peak_sel[mid], self.dwell_times_up[mid], '.', markersize=1)
                plt.xlim((-0.1, 1.1))
                # --- draw intensity cut line
                c_ylim = plt.get(plt.gca(), 'ylim')
                plt.plot([min_val, min_val], [0, c_ylim[1]], 'r')
                plt.plot([max_val, max_val], [0, c_ylim[1]], 'r')
                plt.ylim(0, c_ylim[1])
                plt.title(f'left click for {minormax} intensity cut (enter to move on)')
                plt.xlabel('FRET')
                plt.ylabel('Dwell times(s)')
                plt.show(block=False)
                plt.pause(0.001)
                # --- get mouse input
                user_response = plt.ginput(1, timeout=0)
                if user_response:
                    if minormax == 'min':
                        min_val = user_response[0][0]
                    else:
                        max_val = user_response[0][0]
            return [min_val, max_val]

        e_min_val = 0
        e_max_val = 1
        e_min_val = get_thresh(e_min_val, e_max_val, minormax='min')[0]
        e_max_val = get_thresh(e_min_val, e_max_val, minormax='max')[1]
        plt.close(fignum)
        plt.pause(0.001)

        # --- collect dwells within the FRET range
        dwells_in_range = []
        for dw_times_in_mol, e_in_mol in zip(self.dwell_times_up, self.E_per_peak_sel):
            for tmp_dw_time, tmp_e in zip(dw_times_in_mol, e_in_mol):
                if e_min_val < tmp_e < e_max_val:
                    dwells_in_range.append(tmp_dw_time)

        tmp_down_dwells = [elmt for tmps in self.dwell_times_down for elmt in tmps]

        def fit_exps(data_x, data_y, n_comp):
            def fn_exps(x, *p):  # a, xc, w = p
                _n_peaks = int(len(p) / 3)
                y = 0.0
                for cid in range(n_comp):
                    amp = p[2 * cid]
                    tau = p[2 * cid + 1]
                    y += amp * np.exp(-(x / tau))
                return y

            para_init = []
            para_bounds = []
            del_tau = np.sum([elmtx * elmty for elmtx, elmty in zip(data_x, data_y)]) / np.sum(data_y)
            del_amp = np.max(data_y) / n_comp * 1.5  # 1.2 is a fudge factor to compensate the deletion of the first bin
            for pki in range(n_comp):
                para_init += [del_amp, (del_tau / n_comp) * (pki + 1) ** 2]
                para_bounds += [np.Inf, np.Inf]
            para_bounds = ([0.0, 0.0] * n_comp, para_bounds)
            try:
                coeff, var_matrix = optimize.curve_fit(fn_exps, data_x, data_y, p0=para_init, bounds=para_bounds,
                                                       maxfev=80000)
                fc_x_vct = data_x
                fc = [fc_x_vct] + [fn_exps(fc_x_vct, *coeff)]
                amps = []
                taus = []
                for pki in range(n_comp):
                    amps.append(coeff[pki * 2 + 0])
                    taus.append(coeff[pki * 2 + 1])
            except:
                fc, amps, taus = [], [], []

            return fc, amps, taus

        def get_dwell_hist(entry):
            tmp_t_bin_del = np.mean(entry) / 8
            tmp_t_bin = [tmp_t_bin_del * i for i in range(50)]
            dwell_hist, bin_edges = np.histogram(entry, tmp_t_bin)
            delta_bin = bin_edges[1] - bin_edges[0]
            bin_edges = [elmt + delta_bin for elmt in bin_edges[1:-1]]
            dwell_hist = [bin_edges] + [dwell_hist[1:]]
            return dwell_hist

        dwell_hist_down = get_dwell_hist(tmp_down_dwells)
        dwell_hist_up = get_dwell_hist(dwells_in_range)

        dwell_hist_down_fit1 = fit_exps(dwell_hist_down[0], dwell_hist_down[1], n_comp=1)
        dwell_hist_down_fit2 = fit_exps(dwell_hist_down[0], dwell_hist_down[1], n_comp=2)
        dwell_hist_up_fit1 = fit_exps(dwell_hist_up[0], dwell_hist_up[1], n_comp=1)
        dwell_hist_up_fit2 = fit_exps(dwell_hist_up[0], dwell_hist_up[1], n_comp=2)

        # --- draw histograms
        fhd_dwell_hist = plt.figure(fignum, figsize=(3, 5))
        plt.clf()
        gs1 = fhd_dwell_hist.add_gridspec(nrows=2, ncols=1, left=0.25, right=0.9, bottom=0.1, hspace=0.35)
        plt.subplot(gs1[0])
        plt.bar(dwell_hist_down[0], dwell_hist_down[1], width=dwell_hist_down[0][1] - dwell_hist_down[0][0],
                linewidth=0, facecolor='grey', align='center')
        plt.plot(dwell_hist_down_fit1[0][0], dwell_hist_down_fit1[0][1], color='red', label='1 exp')
        plt.text(dwell_hist_down_fit1[2][0] * 1.5, dwell_hist_down_fit1[1][0] / 3,
                 f'tau={dwell_hist_down_fit1[2][0]:.3g}')
        if dwell_hist_down_fit2[0]:
            plt.plot(dwell_hist_down_fit2[0][0], dwell_hist_down_fit2[0][1], color='blue', label='2 exp')
            plt.text(dwell_hist_down_fit1[2][0] * 2.5, dwell_hist_down_fit1[1][0] / 5,
                     f'tau={dwell_hist_down_fit2[2][0]:.3g} & {dwell_hist_down_fit2[2][1]:.3g}')
        plt.title(f'FRET {e_min_val:.3f}~{e_max_val:.3f}')
        plt.legend(frameon=False)
        plt.xlabel('Down dwell times (s)')
        plt.ylabel('# of binding events')

        plt.subplot(gs1[1])
        plt.bar(dwell_hist_up[0], dwell_hist_up[1], width=dwell_hist_up[0][1] - dwell_hist_up[0][0],
                linewidth=0, facecolor='orange', align='center')
        plt.plot(dwell_hist_up_fit1[0][0], dwell_hist_up_fit1[0][1], color='red', label='1 exp')
        plt.text(dwell_hist_up_fit1[2][0] * 1.5, dwell_hist_up_fit1[1][0] / 3, f'tau={dwell_hist_up_fit1[2][0]:.3g}')
        if dwell_hist_up_fit2[0]:
            plt.plot(dwell_hist_up_fit2[0][0], dwell_hist_up_fit2[0][1], color='blue', label='2 exp')
            plt.text(dwell_hist_up_fit1[2][0] * 2.5, dwell_hist_up_fit1[1][0] / 5,
                     f'tau={dwell_hist_up_fit2[2][0]:.3g} & {dwell_hist_up_fit2[2][1]:.3g}')
        plt.legend(frameon=False)
        plt.xlabel('Up dwell times (s)')
        plt.ylabel('# of binding events')

        plt.show(block=False)
        plt.pause(0.001)

        return fhd_dwell_hist

    def show_trace(self, fignum, rene_id, output_file_path):
        def draw_traces():
            mid = self.c_mid

            # --- draw kymo
            self.fhd_kymograph_peak_idv, self.btn_fit_idv = self.show_hist_kymo(fignum, rene_id, mid, output_file_path)

            # --- draw time trace
            t_vct = [i * self.time_res for i in range(self.trace_len)]
            plt.sca(ax[0])
            plt.cla()
            plt.plot(t_vct, self.It[mid], color=[.8, .8, .8])
            for pki in range(len(self.I_in_peak_sel[mid])):
                tmp_t_vct = [(self.trans_d2u_sel[mid][pki] + i + 1) * self.time_res for i in
                             range(len(self.I_in_peak_sel[mid][pki]))]
                plt.plot(tmp_t_vct, self.I_in_peak_sel[mid][pki], 'k')
            plt.xlim(0, self.trace_len * self.time_res)
            plt.ylim(self.Int_min, self.Int_max)
            plt.ylabel('Intensity (a.u.)')
            plt.title(f'mid={mid}')
            plt.tick_params(axis='x', bottom=True, top=True, labeltop=True, labelbottom=False)

            plt.sca(ax[1])
            plt.cla()
            plt.plot(t_vct, self.Id[mid], color=[.6, 1, .6])
            plt.plot(t_vct, self.Ia[mid], color=[1, .6, .6])
            for pki in range(len(self.I_in_peak_sel[mid])):
                tmp_t_vct = [(self.trans_d2u_sel[mid][pki] + i + 1) * self.time_res for i in
                             range(len(self.I_in_peak_sel[mid][pki]))]
                tmp_d_vct = self.Id[mid][self.trans_d2u_sel[mid][pki] + 1:(
                        self.trans_d2u_sel[mid][pki] + 1 + len(self.I_in_peak_sel[mid][pki]))]
                tmp_a_vct = self.Ia[mid][self.trans_d2u_sel[mid][pki] + 1:(
                        self.trans_d2u_sel[mid][pki] + 1 + len(self.I_in_peak_sel[mid][pki]))]
                plt.plot(tmp_t_vct, tmp_d_vct, color=[0, .5, 0])
                plt.plot(tmp_t_vct, tmp_a_vct, color=[.5, 0, 0])
            plt.xlim(0, self.trace_len * self.time_res)
            plt.ylim(self.Int_min, self.Int_max)
            plt.ylabel('Intensity (a.u.)')
            plt.tick_params(axis='x', bottom=True, top=True, labeltop=False, labelbottom=False)

            plt.sca(ax[2])
            plt.cla()
            for pki in range(len(self.E_in_peak[mid])):
                tmp_t_vct = [(self.trans_d2u[mid][pki] + i + 1) * self.time_res for i in
                             range(len(self.E_in_peak[mid][pki]))]
                plt.plot(tmp_t_vct, self.E_in_peak[mid][pki], color=[0.8, 0.8, 1])
            for pki in range(len(self.E_in_peak_sel[mid])):
                tmp_t_vct = [(self.trans_d2u_sel[mid][pki] + i + 1) * self.time_res for i in
                             range(len(self.E_in_peak_sel[mid][pki]))]
                plt.plot(tmp_t_vct, self.E_in_peak_sel[mid][pki], color=[0, 0, .5])
            plt.ylim(-0.1, 1.1)
            plt.xlim(0, self.trace_len * self.time_res)
            plt.ylabel('FRET')
            plt.xlabel('Time (s)')
            plt.tick_params(axis='x', bottom=True, top=True, labeltop=False, labelbottom=True)

            plt.draw()
            plt.pause(0.001)

        def show_frames():
            def draw_pma(pma_path, text_fr_nb, ax_tmp):
                frame_nb = int(text_fr_nb.text)
                with open(pma_path) as pma_file:
                    pma_width = np.fromfile(pma_file, np.int16, count=1)[0].astype(int)
                    pma_height = np.fromfile(pma_file, np.int16, count=1)[0].astype(int)
                    frame_size = pma_width * pma_height
                    # pma_number_of_frames = int((pma_file_info.st_size - 4) / frame_size)
                    pma_file.seek(4 + (frame_nb * (pma_width * pma_height)), os.SEEK_SET)
                    im = np.reshape(np.fromfile(pma_file, np.uint8, count=frame_size), (pma_width, pma_height))

                    plt.sca(ax_tmp)
                    plt.imshow(im, cmap=self.cmap, vmin=0, vmax=70)

            def mark_mol_pos(mol_d, mol_a, ax_tmp):
                if mol_d[1]:
                    plt.sca(ax_tmp)
                    plt.scatter(mol_d[1], mol_d[2], s=50, edgecolor='y', facecolors='none')
                    plt.scatter(mol_a[1], mol_a[2], s=50, edgecolor='y', facecolors='none')

            def save_frame(ax_image, text_fr_nb):
                frame_nb = int(text_fr_nb.text)
                # --- prepare sub directory
                file_name_base = os.path.basename(self.tr_path[0]).replace('.traces', '')
                output_dir = os.path.dirname(self.tr_path[0])
                output_dir = os.path.join(output_dir, 'selected_tr_' + file_name_base)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_name = f'tr{self.c_mid}_fr{frame_nb}'
                plt.sca(ax_image)
                _fhd = plt.gcf()
                _fhd.savefig(os.path.join(output_dir, output_name + '.eps'), format='eps')

            frame_number = 500
            dir_name_c = os.path.dirname(self.tr_path[0])
            dir_name_p = os.path.dirname(dir_name_c)
            fname_base, _ = os.path.splitext(os.path.basename(self.tr_path[0]))
            # hel_file_number = fname_base.replace('hel', '')
            file_name_pma = fname_base + '.pma'
            file_name_pks = fname_base + '.pks'

            def find_files(dir_name_c, dir_name_p, file_name):
                _available = False
                _full_path = ''
                for fname_tmp in os.listdir(dir_name_c):
                    if fname_tmp == file_name:
                        _available = True
                        _full_path = os.path.join(dir_name_c, file_name)
                        break

                if not _available:
                    for fname_tmp in os.listdir(dir_name_p):
                        if fname_tmp == file_name:
                            _available = True
                            _full_path = os.path.join(dir_name_p, file_name)
                            break

                return _available, _full_path

            pma_available, pma_full_path = find_files(dir_name_c, dir_name_p, file_name_pma)
            pks_available, pks_full_path = find_files(dir_name_c, dir_name_p, file_name_pks)

            cur_mol_pos_d = []
            cur_mol_pos_a = []
            if pma_available:
                self.fhd_pma_image = plt.figure(2352)
                plt.clf()
                ax_image = plt.axes()
                # --- text box for frame number
                ax_fr_nb = plt.axes([0.71, 0.93, 0.08, 0.05])
                text_fr_nb = TextBox(ax_fr_nb, label='', initial='0')
                # --- button to go
                ax_fr_go = plt.axes([0.8, 0.93, 0.08, 0.05])
                self.btn_fr_go = Button(ax_fr_go, 'Go!')
                self.btn_fr_go.on_clicked(lambda event: draw_pma(pma_full_path, text_fr_nb, ax_image))

                # --- button to save
                ax_fr_save = plt.axes([0.9, 0.93, 0.08, 0.05])
                self.btn_fr_save = Button(ax_fr_save, 'save')
                self.btn_fr_save.on_clicked(lambda event: save_frame(ax_image, text_fr_nb))

                draw_pma(pma_full_path, text_fr_nb, ax_image)

                if pks_available:
                    with open(pks_full_path) as pks_file:
                        mol_pos = np.loadtxt(pks_file)
                        _, n_col = mol_pos.shape
                        if n_col == 3:
                            # generated by python (amazing software)
                            mol_pos_d = mol_pos[::2]
                            mol_pos_a = mol_pos[1::2]
                        else:
                            # generated by IDL (This has extra column for background
                            mol_pos_d = mol_pos[::2]
                            mol_pos_a = mol_pos[1::2]

                        cur_mol_pos_d = mol_pos_d[self.c_mid]
                        cur_mol_pos_a = mol_pos_a[self.c_mid]

                        mark_mol_pos(cur_mol_pos_d, cur_mol_pos_a, ax_image)
            plt.show(block=False)
            plt.pause(0.1)

        def show_molecule_x(direction=1):
            self.c_mid = int(text_mid.text)
            self.c_mid += direction
            self.update_log(f'drawing molecule #{self.c_mid}... this may take a while.')
            self.set_progressbar(5)

            text_mid.set_val(str(self.c_mid))
            plt.show(block=False)
            plt.pause(0.1)
            draw_traces()
            show_frames()

            self.update_log(f'Trace updated for molecule #{self.c_mid}.')
            self.set_progressbar(100)

        def save_trace(event):
            mid = self.c_mid
            # --- prepare sub directory
            file_name_base = os.path.basename(self.tr_path[0]).replace('.traces', '')
            output_dir = os.path.dirname(self.tr_path[0])
            output_dir = os.path.join(output_dir, 'selected_tr_' + file_name_base)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # --- save trace
            t_vct = [i * self.time_res for i in range(self.trace_len)]
            output_array = [[i, j, k] for i, j, k in zip(t_vct, self.Ia[mid], self.Id[mid])]
            np.savetxt(os.path.join(output_dir, f'tr{mid}_trace.dat'), output_array)

            # --- save dwell time info
            tmp_dw_t = [len(elmts) * self.time_res for elmts in self.I_in_peak_sel[mid]]
            outout_array = [[i, j, k] for i, j, k in zip(self.trans_d2u_sel[mid], tmp_dw_t, self.E_per_peak_sel[mid])]
            np.savetxt(os.path.join(output_dir, f'tr{mid}_dw.dat'), outout_array)

            # --- save the figure
            self.fhd_trace.savefig(os.path.join(output_dir, f'tr{mid}_trace.eps'), format='eps')

            # --- move on to the next molecule
            # show_molecule_x(direction=1)

            # --- save kymo
            self.fhd_kymograph_peak_idv.savefig(os.path.join(output_dir, f'tr{mid}_kymo.eps'), format='eps')

        self.fhd_trace = plt.figure(fignum, figsize=(13, 5))
        plt.clf()
        gs1 = self.fhd_trace.add_gridspec(nrows=3, ncols=1, left=0.15, right=0.9, bottom=0.1, top=0.8, hspace=0.1)
        ax = [plt.subplot(gs1[0]), plt.subplot(gs1[1]), plt.subplot(gs1[2])]
        # --- button to previous
        ax_btn_prev = plt.axes([0.6, 0.93, 0.08, 0.05])
        self.btn_prev = Button(ax_btn_prev, 'Prev.')
        self.btn_prev.on_clicked(lambda event: show_molecule_x(-1))
        # --- button to next
        ax_btn_next = plt.axes([0.8, 0.93, 0.08, 0.05])
        self.btn_next = Button(ax_btn_next, 'Next.')
        self.btn_next.on_clicked(lambda event: show_molecule_x(1))
        # --- text box for mid
        ax_btn_next = plt.axes([0.71, 0.93, 0.08, 0.05])
        text_mid = TextBox(ax_btn_next, label='', initial='0')
        # text_mid.on_submit(change_mid)
        # --- button to save
        ax_btn_save = plt.axes([0.91, 0.93, 0.08, 0.05])
        self.btn_save = Button(ax_btn_save, 'save')
        self.btn_save.on_clicked(save_trace)

        self.c_mid = 0
        draw_traces()
        show_frames()


class RenePlotTS(RenePlotSimple):
    def show_hist_kymo_ts(self, ax):
        color_code_dark, color_code_bright = gen_color_code(nx=self.N_trace, ny=int(self.trace_len / 3))
        plt.sca(ax)
        # --- draw individual binding traces
        e_datas = self.E_in_peak_sel
        t_datas = self.trans_d2u_sel
        for tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes in zip(e_datas, t_datas, color_code_bright):
            for tmp_etrain, tmp_t_start, c_code in zip(tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes):
                tmp_fr_vct = [(tmp_t_start + elmt) * self.time_res for elmt in range(len(tmp_etrain))]
                plt.plot(tmp_etrain, tmp_fr_vct, color=c_code, linewidth=0.5)
        # --- draw average FRET values
        for tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes in zip(self.E_per_peak_sel, self.trans_d2u_sel,
                                                                       color_code_dark):
            for tmp_e_value, tmp_t_start, c_code in zip(tmp_E_in_peak_per_mol, tmp_t_start_per_mol, c_codes):
                plt.plot(tmp_e_value, tmp_t_start * self.time_res, '.', color=c_code, markersize=2)
        plt.ylabel('Time (s)')

    def show_ehist_ts(self, ax):
        # start curve fit
        n_peaks = 2
        barcodes_pos = []
        barcodes_weight = []
        barcodes_width = []

        x_data = np.add(self.e_bin[0:-1], (self.e_bin[2] - self.e_bin[1]) / 2)
        fc_x_vct = np.arange(x_data[0], x_data[-1], (x_data[1] - x_data[0]) / 10)
        tmp_y = self.peak_hist

        def fn_gauss2(x, *p):  # a, xc, w = p
            _n_peaks = int(len(p) / 3)
            y = 0.0
            for pid in range(_n_peaks):
                y += np.exp(-(x - p[pid * 3 + 1]) ** 2 / p[pid * 3 + 2] ** 2 / 2) * p[pid * 3] / p[
                    pid * 3 + 2] / np.sqrt(2 * np.pi)
            return y

        para_init = []
        para_bounds = []
        coords = self.barcode_pos_peak
        for pki in range(n_peaks):
            para_init += [25.0, coords[pki], 0.05]
            para_bounds += [np.Inf, 1.1, 0.05]
        para_bounds = ([0.0, 0.0, 0.0] * n_peaks, para_bounds)
        coeff, var_matrix = optimize.curve_fit(fn_gauss2, x_data, tmp_y, p0=para_init, bounds=para_bounds)
        fc = fn_gauss2(fc_x_vct, *coeff)

        for pki in range(n_peaks):
            barcodes_weight.append(coeff[pki * 3 + 0])
            barcodes_pos.append(coeff[pki * 3 + 1])
            barcodes_width.append(coeff[pki * 3 + 2])

        # update user data
        self.barcode_pos_peak = barcodes_pos

        # updata figures
        self.update_histogram(ax)
        plt.sca(ax)
        plt.plot(fc_x_vct, fc, 'r')


class RenePlotKimohist(Rene):
    def show_hist_kymo(self, fig_hdl, trace_id=0, n_traces=3, mid=-1, draw_as_peak=True):
        if draw_as_peak:
            if mid == -1:  # plot all the molecules
                c_hist_kymo = self.peak_hist_kymo
                hist_a = self.peak_hist
                hist_centers = self.barcode_pos_peak
            else:  # plot specific molecule
                c_hist_kymo = self.peak_hist_kymo_idv[mid, :, :]
                hist_a = self.peak_hist_idv[mid, :]
                hist_centers = self.barcode_pos_peak_idv[mid]
        else:
            if mid == -1:  # plot all the molecules
                c_hist_kymo = self.hist_kymo
                hist_a = self.hist
                hist_centers = self.barcode_pos_datapt
            else:  # plot specific molecule
                c_hist_kymo = self.hist_kymo_idv[mid, :, :]
                hist_a = self.hist_idv[mid, :]
                hist_centers = self.barcode_pos_datapt_idv[mid]

        self.draw_kymo(fig_hdl, trace_id, n_traces, c_hist_kymo, hist_a, hist_centers, mid)

    def draw_kymo(self, fighdl, trace_id, n_traces, c_hist_kymo, hist_a, hist_centers, mid):
        n_line_kimo, n_e_bin = np.shape(self.hist_kymo)
        t_grid, e_grid = np.mgrid[0:n_line_kimo, self.e_bin[0]:self.e_bin[-1]:(self.e_bin[1] - self.e_bin[0])]
        t_grid *= self.time_res * self.t_bin_size

        # draw hist kymograph
        fighdl.ax[trace_id].pcolor(e_grid, t_grid, c_hist_kymo, cmap='gray',
                                   vmin=c_hist_kymo.min(),
                                   vmax=c_hist_kymo.max() / 2)
        dummy_y = [0, t_grid[-1][0]]
        # draw barcode on the kymo
        for c_cen in hist_centers:
            dummy_x = [c_cen, c_cen]
            fighdl.ax[trace_id].plot(dummy_x, dummy_y, color=(1, .8, 0), lw=3)
        if mid != -1:
            # draw detected barcode with diff. color
            for bid in self.barcodes_detected_ind[mid]:
                dummy_x = [self.barcode_pos_peak[bid], self.barcode_pos_peak[bid]]
                fighdl.ax[trace_id].plot(dummy_x, dummy_y, color=(1, 0, 0), lw=3)

        fighdl.ax[trace_id].set_ylabel('Time (s)')
        fighdl.ax[trace_id].set_ylim(t_grid[-1][0], 0)
        if trace_id == 0:
            fighdl.ax[trace_id].set_title(f'mol #{mid}')
            fighdl.ax[trace_id].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,  # ticks along the bottom edge are off
                top=True,  # ticks along the top edge are off
                labeltop=True,
                labelbottom=False)  # labels along the bottom edge are off
        else:
            fighdl.ax[trace_id].tick_params(
                axis='x',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,  # ticks along the bottom edge are off
                top=True,  # ticks along the top edge are off
                labeltop=False,
                labelbottom=False)  # labels along the bottom edge are off

        # draw barcode
        for c_cen in hist_centers:
            dummy_x = [c_cen, c_cen]
            fighdl.ax[n_traces].plot(dummy_x, [0, 1], color=(0, 0, 0), lw=5)
        fighdl.ax[n_traces].set_xlim((e_grid[0][0], e_grid[-1][-1]))
        fighdl.ax[n_traces].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=True,  # ticks along the bottom edge are off
            top=True,  # ticks along the top edge are off
            labeltop=False,
            labelbottom=False)  # labels along the bottom edge are off
        fighdl.ax[n_traces].tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False,
            labelright=False)  # labels along the bottom edge are off

        # draw histograms
        fighdl.ax[n_traces + 1].bar(e_grid[1, :], hist_a, self.e_bin[1] - self.e_bin[0])
        fighdl.ax[n_traces + 1].set_xlim((e_grid[0, 0], e_grid[0, -1]))
        fighdl.ax[n_traces + 1].set_xlabel('FRET')

    def show_trace(self, fighdl, trace_id=0, n_rene=1, mid=0):
        # plot intensity traces
        ax = fighdl.ax[0][trace_id]
        ax.plot(self.It[mid], color=(.7, .7, .7))
        ax.plot(self.It_ideal[mid], 'k')
        ax.plot(self.Id[mid], 'g-')
        ax.plot(self.Ia[mid], 'r-')
        ax.grid(True)
        ax.set_ylim((-100, 1000))
        ax.set_xlim((0, self.trace_len))

        if trace_id == 0:
            ax.set_ylabel('Fluorescence (a.u.)')
            ax.set_title(f'mol #{mid}')
        else:
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=True,  # ticks along the bottom edge are off
                right=True,  # ticks along the top edge are off
                labelleft=False,
                labelright=False)  # labels along the bottom edge are off

        # plot FRET trace
        ax = fighdl.ax[1][trace_id]
        ax.plot(self.E[mid], color=(.9, .9, 1))
        # ax.plot(self.E_clean[mid], 'b-')

        tmp_e_trains = self.E_in_peak[trace_id]
        tmp_n_peak = len(tmp_e_trains)
        for pid in range(tmp_n_peak):
            tmp_e_train = tmp_e_trains[pid]
            tmp_t_vct = [self.trans_d2u[trace_id][pid] + elmt for elmt in range(len(tmp_e_train))]
            ax.plot(tmp_t_vct, tmp_e_train)

        ax.grid(True)
        ax.set_ylim((-0.1, 1.1))
        ax.set_xlim((0, self.trace_len))

        if trace_id == 0:
            ax.set_ylabel('FRET')
        else:
            ax.tick_params(
                axis='y',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                left=True,  # ticks along the bottom edge are off
                right=True,  # ticks along the top edge are off
                labelleft=False,
                labelright=False)  # labels along the bottom edge are off
