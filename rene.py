import sys
import time
import numpy as np
import os
from sys import platform
# if platform == "darwin":
from matplotlib import use
use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import optimize


def run_kmeans(data_train, n_cluster=2, min_cluster_dist=36.):
    k_train = np.reshape(data_train, (-1, 1))
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(k_train)
    levels = kmeans.predict(k_train)
    centroids = []
    for i in range(n_cluster):
        centroids.append(kmeans.cluster_centers_[i][0])

    # For 2-states check if the first level is the lowest FRET state
    if n_cluster == 2:
        if centroids[1] < centroids[0]:
            # swap the states
            centroids = [centroids[1], centroids[0]]
            tmp_levels = levels * 0
            for i, level in enumerate(levels):
                if level == 0:
                    tmp_levels[i] = 1
            levels = tmp_levels

        # accept only if the two classes are significantly far apart (min_cluster_dist x Allan var)
        allan_var = np.average(np.square(np.subtract(data_train[2:], data_train[1:-1]))) / 2
        cent_dist = centroids[1] - centroids[0]
        if cent_dist * cent_dist < allan_var * min_cluster_dist:
            levels = np.multiply(levels, 0)
            centroids[1] = centroids[0]

    return [levels, centroids]


def fn_gauss_multi(x, *p):  # a, xc, w = p
    _n_peaks = int(len(p) / 3)
    y = 0.0
    for pid in range(_n_peaks):
        y += np.exp(-(x - p[pid * 3 + 1]) ** 2 / p[pid * 3 + 2] ** 2 / 2) * p[pid * 3] / p[pid * 3 + 2] / np.sqrt(2 * np.pi)
    return y


class Barcode:
    def __init__(self, datatype='peak'):
        self.datatype = datatype
        self.pos = []
        self.weight = []
        self.weight_fraction = []
        self.std = []
        self.sem = []
        self.fc = []
        self.gmm_set = {
            "n_cluster": 5,
            "min_occupancy": 0.1,
            "min_distance": 0.02}
        self.gmm_tot_data_points = []
        self.gmm_data = []

    def run_gmm(self, data_train, n_cluster=2, min_occupancy=0.1, min_distance=0.1):
        if data_train:
            np_train = np.asarray(data_train)
            x = np_train.reshape(-1, 1)

            models = GaussianMixture(n_cluster).fit(x)
            centroids = np.squeeze(models.means_)
            widths = np.squeeze(models.covariances_)
            weights = models.weights_

            # sort by centroids (increase)
            dtype = [('centroids', float), ('widths', float), ('weights', float), ('weights_r', float)]
            gmm_data = []
            for i in range(n_cluster):
                gmm_data.append((centroids[i], widths[i], weights[i], weights[i]))
            gmm_data = np.array(gmm_data, dtype=dtype)
            gmm_data = np.sort(gmm_data, order='centroids')

            # recalculate weights without d-only peak (weight_r)
            weights_r = []
            for i in range(n_cluster):
                if i != 0:
                    weights_r.append(gmm_data[i][3])
            weights_r = [elmt / (sum(weights_r)) for elmt in weights_r]
            weights_r.insert(0, 1)
            for i in range(n_cluster):
                gmm_data[i][3] = weights_r[i]

            # remove peaks of small population
            gmm_data = [elmt for elmt in gmm_data if elmt[2] > min_occupancy]

            # remove peaks too close each other
            gmm_data2 = [gmm_data[0]]
            for i in range(1, gmm_data.__len__()):
                if (gmm_data2[-1][0] - gmm_data[i][0]) ** 2 < min_distance ** 2:
                    # update previous peak by averaging with this peak
                    tmpnormf = gmm_data2[-1][2] + gmm_data[i][0]
                    gmm_data2[-1][0] = (gmm_data2[-1][0] * gmm_data2[-1][2] / tmpnormf + gmm_data[i][0] * gmm_data[i][
                        0] / tmpnormf)  # weighted mean of the FRET values
                    gmm_data2[-1][1] = gmm_data2[-1][1] + gmm_data[i][1]  # covariances are addable
                    gmm_data2[-1][2] = gmm_data2[-1][2] + gmm_data[i][2]  # weights are addabel
                    gmm_data2[-1][3] = gmm_data2[-1][3] + gmm_data[i][3]
                else:
                    gmm_data2.append(gmm_data[i])
        else:
            gmm_data2 = []

        return gmm_data2

    def build_gmm_fc(self, fc_x, density_factor):
        gmm_fc = [fc_x]
        coef = []
        for elmt in self.gmm_data:
            coef.append(elmt[2] * (gmm_fc[0][1]-gmm_fc[0][0]) * self.gmm_tot_data_points * density_factor)
            coef.append(elmt[0])
            coef.append(np.sqrt(elmt[1]))
        coef = np.array(coef)
        gmm_fc.append(fn_gauss_multi(gmm_fc[0], *coef))
        return gmm_fc

    def find_barcode_by_curve_fit(self, hist_bin, hist_count, raw_data=(), init_pts=()): # either (only either) start_pts or raw_data should be given
        if not init_pts:
            if len(raw_data) <= 1:
                self.gmm_data = []
                self.gmm_tot_data_points = len(raw_data)
                init_pts = raw_data
            else:
                if 1 < len(raw_data) < self.gmm_set['n_cluster']:
                    n_clusters = len(raw_data)
                if len(raw_data) >= self.gmm_set['n_cluster']:
                    n_clusters = self.gmm_set['n_cluster']

                self.gmm_data = self.run_gmm(data_train=raw_data, n_cluster=n_clusters,
                             min_occupancy=self.gmm_set['min_occupancy'], min_distance=self.gmm_set['min_distance'])
                init_pts = [elmt[0] for elmt in self.gmm_data]
                self.gmm_tot_data_points = len(raw_data)
        else:
            self.gmm_data = []
            self.gmm_tot_data_points = 0

        n_peaks = len(init_pts)
        if n_peaks != 0:
            para_init = []
            para_bounds = []
            for pki in range(n_peaks):
                para_init += [max(hist_count) * (hist_bin[1] - hist_bin[0]), init_pts[pki], 0.02]
                para_bounds += [np.Inf, 1.1, 0.05]
            para_bounds = ([0.0, 0.0, 0.0] * n_peaks, para_bounds)
            fit_success = True
            try:
                coeff, var_matrix = optimize.curve_fit(fn_gauss_multi, hist_bin, hist_count, p0=para_init, bounds=para_bounds, maxfev=500)
            except:
                fit_success = False

            density_factor = 10
            self.fc = [np.arange(hist_bin[0], hist_bin[-1], (hist_bin[1] - hist_bin[0]) / density_factor)]

            if fit_success:
                self.fc.append(fn_gauss_multi(self.fc[0], *coeff))
                self.pos = []
                self.weight = []
                self.std = []
                for pki in range(n_peaks):
                    self.weight.append(coeff[pki * 3 + 0] / (hist_bin[1] - hist_bin[0]))
                    self.pos.append(coeff[pki * 3 + 1])
                    self.std.append(coeff[pki * 3 + 2])
                norm_factor = np.sum(self.weight)
                self.weight_fraction = [elmt / norm_factor for elmt in self.weight]
                self.sem = [w/np.sqrt(n) for w, n in zip(self.std, self.weight)]
            else:
                self.fc = self.build_gmm_fc(fc_x=self.fc[0], density_factor=density_factor)
                self.pos = []
                self.weight = []
                self.weight_fraction = []
                self.std = []
                self.sem = []
                for tmp_gmm_data in self.gmm_data:
                    self.weight.append(tmp_gmm_data[2] * self.gmm_tot_data_points)
                    self.weight_fraction.append(tmp_gmm_data[2])
                    self.pos.append(tmp_gmm_data[0])
                    self.std.append(np.sqrt(tmp_gmm_data[1]))
                    self.sem.append(np.sqrt(tmp_gmm_data[1])/np.sqrt(tmp_gmm_data[2] * self.gmm_tot_data_points))
        # else:
        #     self.fc = []
        #     self.pos = []
        #     self.weight = []
        #     self.weight_fraction = []
        #     self.std = []
        #     self.sem = []

        # plt.figure(1113)
        # plt.clf()
        # plt.plot(hist_bin, hist_count, 'k')
        # plt.plot(self.fc[0], self.fc[1], 'r')
        # plt.title(fit_success)
        # plt.show(block=False)
        # plt.pause(0.1)


class Rene:
    def __init__(self, trace_path, uip=None, pg_bar=None, log_txt=None, gui_obj=None):
        self.log_txt = log_txt
        self.pg_bar = pg_bar
        self.tr_path = trace_path
        self.gui_obj = gui_obj

        # user-defined parameters
        if uip is None:
            uip = {
                "time_res": 0.1,
                "gamma": 1.0,
                "leakage": 0.0,
                "bg_d_fix": 0,
                "bg_a_fix": 0,
                "Int_max": 1000,
                "Int_min": -100,

                "keyword": "blink",
                "min_peak_len": 0.3,
                "min_peak_int": 0,
                "max_peak_int": float('Inf'),
                "E_tolerance": 0.2,

                "t_bin_size": 50,
                "e_bin_size": 0.01,
            }

        # print(f'Working on {os.path.basename(self.tr_path[0])}...')

        self.set_parameters(uip, run_afterwards=False)

        # dependent parameters
        self.e_bin = np.arange(-0.1, 1.1, self.e_bin_size)
        self.trace_len = 0
        self.N_trace = 0
        self.bg_d = []
        self.bg_a = []

        self.Id_org = list()
        self.Ia_org = list()
        self.Id = list()
        self.Ia = list()
        self.It = list()
        self.It_ideal = list()
        self.It_level = list()
        self.It_ideal_centroids = list()
        self.E = list()
        self.E_clean = list()
        self.trans_d2u = []
        self.trans_u2d = []

        self.hist = []
        self.hist_idv = []
        self.hist_kymo = []
        self.hist_kymo_idv = []

        self.E_per_peak = []
        self.E_in_peak = []

        self.run_analysis()

    def set_progressbar(self, c_progress):
        if self.pg_bar:
            self.pg_bar["value"] = c_progress
            self.pg_bar.update()

    def update_log(self, logtxt):
        if self.log_txt:
            self.log_txt.set(logtxt)
        else:
            sys.stdout.write(logtxt)

    def set_parameters(self, uip=None, run_afterwards=False):
        self.uip = uip

        # self.uip['time_res'] = uip["time_res"]
        self.gamma = uip["gamma"]
        self.leakage = uip["leakage"]
        # self.bg_d_fix = uip["bg_d_fix"]
        # self.bg_a_fix = uip["bg_a_fix"]
        self.Int_max = uip["Int_max"]
        self.Int_min = uip["Int_min"]
        self.toi_start = uip["toi_start"]
        self.toi_end = uip["toi_end"]

        self.keyword = uip["keyword"]
        self.min_peak_len = uip["min_peak_len"]
        # self.uip['min_peak_int'] = uip["min_peak_int"]
        # self.uip['max_peak_int'] = uip["max_peak_int"]
        self.E_tolerance = uip["E_tolerance"]

        self.t_bin_size = uip["t_bin_size"]
        self.e_bin_size = uip["e_bin_size"]

        self.cmap = uip["cmap"]
        # if run_afterwards:
        #     self.run_analysis()

    def run_analysis(self):
        t0 = time.time()
        self.update_log(self.tr_path[0]+'\n')

        self.set_progressbar(1)
        self.update_log(f"\r   getting trace files...")
        self.read_traces_file()
        self.set_progressbar(5)
        self.update_log(f"\r   peak finding...")
        self.find_peak()
        self.set_progressbar(20)
        self.update_log(f"\r   adjusting backgrounds...")
        self.auto_bg_set()
        self.set_progressbar(25)
        self.update_log(f"\r   calculating FRET...")
        self.calc_fret()
        self.set_progressbar(30)
        self.update_log(f"\r   building most likely intensity traces...")
        self.build_it_ideal()
        self.set_progressbar(40)
        self.update_log(f"\r   collecting dwell time info...")
        self.analyze_peaks()
        self.set_progressbar(50)
        self.update_log(f"\r   building histograms...")
        self.build_hist_kymo()
        self.update_log(f"\r   make barcodes...")
        self.determine_barcodes()

        self.set_progressbar(100)
        t1 = time.time()
        elapsed_time = (t1 - t0) * 1000 // 1000

        self.update_log(f"\r   Job done. (served in {elapsed_time}s)\n")

    def read_traces_file(self):
        tmp_d = []
        tmp_a = []
        self.trace_len = 2**32  # just a big int number
        self.N_trace = 0
        for c_pth in self.tr_path:
            _, f_ext = os.path.splitext(c_pth)
            if 'traces' in f_ext:
                # load binary file
                with open(c_pth, 'rb') as trf_obj:
                    self.trace_len = int(np.fromfile(trf_obj, dtype=np.int32, count=1))
                    if f_ext == '.traces2':
                        self.N_trace = int(np.fromfile(trf_obj, dtype=np.int32, count=1) / 2)
                    else:
                        self.N_trace = int(np.fromfile(trf_obj, dtype=np.int16, count=1) / 2)
                    raw_data = np.fromfile(trf_obj, dtype=np.int16)
                    raw_data2 = raw_data.reshape((self.trace_len, self.N_trace * 2))
                    raw_data2 = np.transpose(raw_data2)
                    tmp_d = raw_data2[0::2][:].astype(float)
                    tmp_a = raw_data2[1::2][:].astype(float)
            else:
                # load aschii files
                self.N_trace += 1
                raw_d = []
                raw_a = []
                c_len = 0
                with open(c_pth) as trf_obj:
                    for line in trf_obj:
                        c_len += 1
                        tmpstr = line.replace('\n', '')
                        tmpstr2 = tmpstr.split()
                        raw_d.append(float(tmpstr2[1]))
                        raw_a.append(float(tmpstr2[2]))
                tmp_d.append(np.array(raw_d))
                tmp_a.append(np.array(raw_a))
                if c_len < self.trace_len:
                    self.trace_len = c_len

        if np.isinf(self.toi_end):
            self.toi_end = self.trace_len
        self.toi_end = int(self.toi_end)
        self.toi_start = int(self.toi_start)
        self.trace_len = self.toi_end - self.toi_start
        self.Id_org = np.zeros((self.N_trace, self.trace_len))
        self.Ia_org = np.zeros((self.N_trace, self.trace_len))
        for mid in range(self.N_trace):
            self.Id_org[mid, :] = tmp_d[mid][self.toi_start:self.toi_end]
            self.Ia_org[mid, :] = tmp_a[mid][self.toi_start:self.toi_end]

    def find_peak(self, algorithm='KMeans'):
        self.It_level = np.zeros((self.N_trace, self.trace_len))
        self.It_ideal_centroids = np.zeros((self.N_trace, 2))
        for mid in range(self.N_trace):
            # self.update_log(f"\r   peak finding... ({mid / self.N_trace * 10000 // 10 / 10}%)")
            if mid % 7 == 0:
                self.update_log(f"\r   peak finding... (tr{mid}/{self.N_trace})")
            self.set_progressbar(5 + (mid * 15 // self.N_trace))
            self.It_level[mid][:], self.It_ideal_centroids[mid][:] = \
                run_kmeans(data_train=self.Id_org[mid] + self.Ia_org[mid])

    def auto_bg_set(self):
        if self.uip["bg_d_fix"] != 0 and self.uip["bg_a_fix"] != 0:
            self.bg_d = [self.uip["bg_d_fix"]] * self.N_trace
            self.bg_a = [self.uip["bg_a_fix"]] * self.N_trace
            print(self.uip["bg_d_fix"])
            print(self.uip["bg_a_fix"])
        else:
            self.bg_d = []
            self.bg_a = []
            for mid in range(self.N_trace):
                if mid % 7 == 0:
                    self.update_log(f"\r   adjusting backgrounds... (tr{mid}/{self.N_trace})")
                    self.set_progressbar(20 + (mid * 10 // self.N_trace) * 2)

                aa = [self.Ia_org[mid][i] for i, elmt in enumerate(self.It_level[mid][:]) if elmt == 0]
                dd = [self.Id_org[mid][i] for i, elmt in enumerate(self.It_level[mid][:]) if elmt == 0]

                aa_threshold = np.average(aa) + np.std(aa) * 2
                aa = [elmt for elmt in aa if elmt < aa_threshold]

                dd_threshold = np.average(dd) + np.std(dd) * 2
                dd = [elmt for elmt in dd if elmt < dd_threshold]

                bg_method = 'average'
                if bg_method == 'average':
                    self.bg_d.append(np.average(dd))
                    self.bg_a.append(np.average(aa))
                elif bg_method == 'bottom gaussian':
                    dummy, d_centroids = run_kmeans(data_train=dd, n_cluster=2, min_cluster_dist=1)
                    dummy, a_centroids = run_kmeans(data_train=aa, n_cluster=2, min_cluster_dist=1)
                    self.bg_d.append(min(d_centroids))
                    self.bg_a.append(min(a_centroids))

    def calc_fret(self):
        self.Id = np.zeros_like(self.Id_org)
        self.Ia = np.zeros_like(self.Ia_org)
        for mid in range(self.N_trace):
            # sys.stdout.write(f"\r   calculating FRET...  ({mid / self.N_trace * 10000 // 10 / 20}%)")
            self.Id[mid][:] = [tmpn - self.bg_d[mid] for tmpn in self.Id_org[mid][:]]
            self.Ia[mid][:] = [tmpn - self.bg_a[mid] for tmpn in self.Ia_org[mid][:]]

        self.Ia = np.multiply(self.Ia, self.gamma)

        self.Id += np.multiply(self.Id, self.leakage)
        self.Ia -= np.multiply(self.Id, self.leakage)

        self.E = np.true_divide(self.Ia, self.Id + self.Ia)
        self.It = self.Ia + self.Id

        # build a clean FRET trace by removing low intensity data
        n_mtag, n_ttag = self.E.shape
        self.E_clean = np.full((n_mtag, n_ttag), np.inf)
        for mid in range(n_mtag):
            for tid in range(n_ttag):
                if self.It_level[mid, tid] != 0:
                    self.E_clean[mid, tid] = self.E[mid, tid]

    def build_it_ideal(self):
        self.It_ideal = np.zeros_like(self.It_level)
        for mid in range(self.N_trace):
            if mid % 7 == 0:
                self.update_log(f"\r   building most likely intensity traces... (tr{mid}/{self.N_trace})")
                self.set_progressbar(35 + (mid * 5 // self.N_trace))
            aa = [self.Ia[mid][i] for i, elmt in enumerate(self.It_level[mid][:]) if elmt == 1]
            dd = [self.Id[mid][i] for i, elmt in enumerate(self.It_level[mid][:]) if elmt == 1]
            if aa:
                icenter = np.average(aa) + np.average(dd)
                self.It_ideal[mid][:] = np.multiply(self.It_level[mid][:], icenter)

    def analyze_peaks(self):
        # --- get transition points
        for mid in range(self.N_trace):
            if mid % 8 == 0:
                self.update_log(f"\r   analyze peaks... (tr{mid}/{self.N_trace})")
                self.set_progressbar(40 + (mid * 10 // self.N_trace))
            tmp_trans = np.diff(self.It_level[mid])
            e_diff = np.abs(np.diff(self.E[mid]))
            e_diff = e_diff * self.It_level[mid][1:] * self.It_level[mid][:-1]

            tmp_d2u = []
            tmp_u2d = []
            is_on = False
            if self.It_level[mid][0] == 1:
                tmp_d2u.append(-1)
                is_on = True
            for i, trans in enumerate(tmp_trans):
                if trans == 1:
                    tmp_d2u.append(i)
                    is_on = True
                elif trans == -1:
                    tmp_u2d.append(i)
                    is_on = False
                elif is_on and (e_diff[i] > self.E_tolerance):
                    # start new peaks
                    tmp_d2u.append(i)
                    tmp_u2d.append(i)

            if self.It_level[mid][-1] == 1:
                tmp_u2d.append(self.trace_len)
            self.trans_d2u.append(tmp_d2u)
            self.trans_u2d.append(tmp_u2d)

        # --- count number of binding events
        n_binding_events = np.zeros((self.N_trace,))
        for mid in range(self.N_trace):
            n_binding_events[mid] = self.trans_d2u[mid][:].__len__()

        # --- get averaged E, I per peak
        self.E_per_peak = [[]] * self.N_trace
        self.E_in_peak = [[]] * self.N_trace
        self.I_per_peak = [[]] * self.N_trace
        self.I_in_peak = [[]] * self.N_trace
        for mid in range(self.N_trace):
            if mid % 7 == 0:
                self.update_log(f"\r   analyze peaks... (tr{mid}/{self.N_trace})")
                self.set_progressbar(70 + (mid * 10 // self.N_trace))
            self.E_per_peak[mid] = [0] * int(n_binding_events[mid])
            self.I_per_peak[mid] = [0] * int(n_binding_events[mid])
            self.E_in_peak[mid] = [0] * int(n_binding_events[mid])
            self.I_in_peak[mid] = [0] * int(n_binding_events[mid])
            I_tot=self.Ia[mid]+self.Id[mid]
            for pki in range(int(n_binding_events[mid])):
                tmp_e_data = self.E[mid][self.trans_d2u[mid][pki] + 1:self.trans_u2d[mid][pki]]
                tmp_i_data = I_tot[self.trans_d2u[mid][pki] + 1:self.trans_u2d[mid][pki]]
                if tmp_e_data.size == 0:
                    self.E_per_peak[mid][pki] = np.nan
                    self.E_in_peak[mid][pki] = np.empty(0)
                    self.I_per_peak[mid][pki] = np.nan
                    self.I_in_peak[mid][pki] = np.empty(0)
                else:
                    self.E_per_peak[mid][pki] = np.average(tmp_e_data)
                    self.E_in_peak[mid][pki] = tmp_e_data
                    self.I_per_peak[mid][pki] = np.average(tmp_i_data)
                    self.I_in_peak[mid][pki] = tmp_i_data
        plt.close('all')
        plt.show(block=True)

        # --- get intensity cutoffs
        fhd_thresh = plt.figure(35271)
        plt.clf()
        ax35271 = plt.axes()

        def get_thresh(max_val, min_val, minormax='min'):
            user_response = True
            while user_response:
                plt.sca(ax35271)
                plt.cla()
                for mid in range(self.N_trace):
                    plt.plot(self.E_per_peak[mid], self.I_per_peak[mid], '.', markersize=1)
                plt.xlim((-0.1, 1.1))
                # draw intensity cut line
                plt.plot([-0.1, 1.1], [min_val, min_val], 'r')
                plt.plot([-0.1, 1.1], [max_val, max_val], 'r')
                c_ylim = plt.get(plt.gca(), 'ylim')
                plt.ylim(0, c_ylim[1])
                plt.title(f'left click for {minormax} intensity cut (enter to move on)')
                plt.show(block=False)
                plt.pause(0.1)
                user_response = plt.ginput(1, timeout=0)
                if user_response:
                    if minormax == 'min':
                        min_val = user_response[0][1]
                    else:
                        max_val = user_response[0][1]
            return [min_val, max_val]

        tmp = get_thresh(self.uip['max_peak_int'], self.uip['min_peak_int'], minormax='min')
        self.uip['min_peak_int'] = int(tmp[0])
        tmp = get_thresh(self.uip['max_peak_int'], self.uip['min_peak_int'], minormax='max')
        self.uip['max_peak_int'] = int(tmp[1])

        if self.gui_obj:
            self.gui_obj.user_parameters(mode='set', uip=self.uip)

        plt.show(block=False)
        plt.close(fhd_thresh)
        plt.pause(0.1)

        # --- find good binding events
        # 1. select those with proper intensity
        vid1 = []
        for mid in range(self.N_trace):
            tmp_vid = [False] * len(self.I_per_peak[mid])
            for pki, tmp_int in enumerate(self.I_per_peak[mid]):
                if self.uip['min_peak_int'] < tmp_int < self.uip['max_peak_int']:
                    tmp_vid[pki] = True
            vid1.append(tmp_vid)

        # 2. select sufficiently longer binding events
        vid2 = []
        for mid in range(self.N_trace):
            tmp_vid = [False] * len(self.I_per_peak[mid])
            for pki, tmp_int in enumerate(self.I_in_peak[mid]):
                if self.min_peak_len <= len(tmp_int) * self.uip['time_res']:
                    tmp_vid[pki] = True
            vid2.append(tmp_vid)

        # combine all the selections
        vid_all = []
        for vid1_per_mol, vid2_per_mol in zip(vid1, vid2):
            vid_all.append([a and b for a, b in zip(vid1_per_mol, vid2_per_mol)])

        # --- select good binding events
        self.I_per_peak_sel = []
        self.E_per_peak_sel = []
        self.I_in_peak_sel = []
        self.E_in_peak_sel = []
        self.trans_d2u_sel = []
        self.trans_u2d_sel = []
        for mid, vid_of_a_mol in enumerate(vid_all):
            tmp_i_per_peak = []
            tmp_e_per_peak = []
            tmp_i_in_peak = []
            tmp_e_in_peak = []
            tmp_d2u = []
            tmp_u2d = []
            for pki, vid_of_a_peak in enumerate(vid_of_a_mol):
                if vid_of_a_peak:
                    tmp_i_per_peak.append(self.I_per_peak[mid][pki])
                    tmp_e_per_peak.append(self.E_per_peak[mid][pki])
                    tmp_i_in_peak.append(self.I_in_peak[mid][pki])
                    tmp_e_in_peak.append(self.E_in_peak[mid][pki])
                    tmp_d2u.append(self.trans_d2u[mid][pki])
                    tmp_u2d.append(self.trans_u2d[mid][pki])
            self.I_per_peak_sel.append(tmp_i_per_peak)
            self.E_per_peak_sel.append(tmp_e_per_peak)
            self.I_in_peak_sel.append(tmp_i_in_peak)
            self.E_in_peak_sel.append(tmp_e_in_peak)
            self.trans_d2u_sel.append(tmp_d2u)
            self.trans_u2d_sel.append(tmp_u2d)

        # --- get dwell time distribution
        self.get_dwell_info()

    def get_dwell_info(self):
        # --- get up dwell times
        self.dwell_times_down = []
        self.dwell_times_up = []
        lid = -1
        for tmp_d2u, tmp_u2d in zip(self.trans_d2u_sel, self.trans_u2d_sel):
            lid += 1
            if tmp_d2u:
                if tmp_d2u[0] != -1:
                    tmp_u2d_c = [0] + tmp_u2d[:-1]
                    tmp_d2u_c = tmp_d2u
                else:
                    tmp_d2u_c = tmp_d2u[1:]
                    tmp_u2d_c = tmp_u2d[:-1]
                self.dwell_times_down.append([(elmt_d2u - elmt_u2d) * self.uip['time_res'] for elmt_d2u, elmt_u2d in zip(tmp_d2u_c, tmp_u2d_c)])
                self.dwell_times_up.append([(elmt_u2d - elmt_d2u) * self.uip['time_res'] for elmt_d2u, elmt_u2d in zip(tmp_d2u, tmp_u2d)])
            else:
                self.dwell_times_down.append([])
                self.dwell_times_up.append([])

    def build_hist_kymo(self):
        self.update_log(f"\r   building hist-kymos...")
        self.set_progressbar(50)

        t_bin = np.arange(0, self.trace_len, self.t_bin_size)
        self.hist_kymo_idv = np.zeros((self.N_trace, len(t_bin) - 1, len(self.e_bin) - 1))
        for i, t_tag in enumerate(t_bin[:-1]):
            for mid in range(self.N_trace):
                tmp_datapt = []
                for j, elmt in enumerate(self.trans_d2u_sel[mid]):
                    if int(t_tag) - 1 <= elmt < (int(t_tag) + self.t_bin_size - 1):
                        tmp_datapt.extend(self.E_in_peak_sel[mid][j])
                tmp_hist, bin_edges = np.histogram(tmp_datapt, bins=self.e_bin)
                self.hist_kymo_idv[mid, i, :] = tmp_hist
        self.hist_kymo = np.sum(self.hist_kymo_idv, axis=0)
        self.hist_idv = np.sum(self.hist_kymo_idv, axis=1)
        self.hist = sum(self.hist_kymo)

        # build hist-kymo from the peak averages
        self.update_log(f"\r   building hist-kymos...")
        self.set_progressbar(55)
        self.peak_hist_kymo_idv = np.zeros_like(self.hist_kymo_idv)
        for i, t_tag in enumerate(t_bin[:-1]):
            for mid in range(self.N_trace):
                tmp_peaks = [self.E_per_peak_sel[mid][j] for j, elmt in enumerate(self.trans_d2u_sel[mid])
                             if int(t_tag) - 1 <= elmt < (int(t_tag) + self.t_bin_size - 1)]
                tmp_hist, bin_edges = np.histogram(tmp_peaks, bins=self.e_bin)
                self.peak_hist_kymo_idv[mid, i, :] = tmp_hist

        self.peak_hist_kymo = np.sum(self.peak_hist_kymo_idv, axis=0)
        self.peak_hist_idv = np.sum(self.peak_hist_kymo_idv, axis=1)
        self.peak_hist = sum(self.peak_hist_kymo)

    def determine_barcodes(self):
        # if mid is not defined, all the molecules will be analyzed
        # --- get peak centers from all data points (Guassian Mixture modeling)
        self.update_log(f"\r   detecting barcode positions...")
        self.set_progressbar(60)

        # find barcode from all molecules
        self.bcd_datapt_all = self.find_barcode_idv(mid=-1, datatype='datapt', init_pts=())
        self.bcd_peak_all = self.find_barcode_idv(mid=-1, datatype='peak', init_pts=())

        # run over each molecule
        self.bcd_datapt_idv = []
        self.bcd_peak_idv = []
        for mid in range(self.N_trace):
            # if mid % 1 == 0:
            self.update_log(f"\r   detecting barcode positions ({mid}/{self.N_trace})...")
            self.set_progressbar(60+mid/self.N_trace*30)

            self.bcd_datapt_idv.append(self.find_barcode_idv(mid=mid, datatype='datapt', init_pts=()))
            self.bcd_peak_idv.append(self.find_barcode_idv(mid=mid, datatype='peak', init_pts=()))

    def find_barcode_idv(self, mid, datatype='peak', init_pts=()):
        tmp_bcd = Barcode(datatype='datapt')

        tmp_input = []
        if mid == -1 and datatype == 'datapt':
            hist_data = self.hist
            for tmp_E_in_mol in self.E_in_peak_sel:
                for tmp_E_in_peak in tmp_E_in_mol:
                    for tmp_Es in tmp_E_in_peak:
                        tmp_input.append(tmp_Es)
        elif mid == -1 and datatype == 'peak':
            hist_data = self.peak_hist
            for tmp_E_peaks in self.E_per_peak_sel:
                for elmt in tmp_E_peaks:
                    tmp_input.append(elmt)
        elif mid is not -1 and datatype == 'datapt':
            hist_data = self.hist_idv[mid]
            for tmp_E_in_peak in self.E_in_peak_sel[mid]:
                for tmp_Es in tmp_E_in_peak:
                    tmp_input.append(tmp_Es)
        elif mid is not -1 and datatype == 'peak':
            hist_data = self.peak_hist_idv[mid]
            for elmt in self.E_per_peak_sel[mid]:
                tmp_input.append(elmt)

        tmp_hist_bin = np.add(self.e_bin[0:-1], (self.e_bin[2] - self.e_bin[1]) / 2)

        # if 1 < len(tmp_input) < self.uip["bcd_max_n_peaks"]:
        #     n_clusters = len(tmp_input)
        # if len(tmp_input) >= self.uip["bcd_max_n_peaks"]:
        #     n_clusters = self.uip["bcd_max_n_peaks"]

        # if 1 < len(tmp_input):
        tmp_bcd.gmm_set = {"n_cluster": self.uip["bcd_max_n_peaks"],
                           "min_occupancy": self.uip["bcd_min_occupancy"],
                           "min_distance": self.uip["bcd_min_distance"]}
        tmp_bcd.find_barcode_by_curve_fit(hist_bin=tmp_hist_bin, hist_count=hist_data, raw_data=tmp_input, init_pts=init_pts)


        # update rene data
        if init_pts:    # if init_pts is not empty, that means this function is manually called for specific mid.
            if mid == -1 and datatype == 'datapt':
                self.bcd_datapt_all = tmp_bcd
            elif mid == -1 and datatype == 'peak':
                self.bcd_peak_all = tmp_bcd
            elif mid is not -1 and datatype == 'datapt':
                self.bcd_datapt_idv[mid] = tmp_bcd
            elif mid is not -1 and datatype == 'peak':
                self.bcd_peak_idv[mid] = tmp_bcd

        return tmp_bcd