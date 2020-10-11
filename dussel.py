import matplotlib
import matplotlib.pyplot as plt
import os.path
import pickle
import sys
import shutil
from rene_plots import RenePlotKimohist
from rene_plots import Fig_kymo_hist
from rene_plots import Fig_trace

graphic_format = 'eps'  # eps or png
# if graphic_format == 'png':
#     matplotlib.use('agg')  # for png file format
# elif graphic_format == 'eps':
#     matplotlib.use('ps')  # for eps file format

data_set = 4
# molecules2show = -1  # set to -1 for all molecules
molecules2show = range(20)

root_path = r'/Users/sunghyunkim/Desktop/true single/'
tr_path = list()
if data_set == 0:
    # branch 1-3
    daily_path = r'20200312 true single/'
    tr_path.append(
        root_path + daily_path + r'1.1 100pM brachded 1.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 1/hel137.traces')
    tr_path.append(
        root_path + daily_path + r'1.2 100pM brachded 1.3 + 10nM MF85 (barcode B) + 10nM MF73 (acceptor) round 2/hel141.traces')
    tr_path.append(
        root_path + daily_path + r'1.3 100pM brachded 1.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 3/hel144.traces')
elif data_set == 1:
    # branch 2-3
    daily_path = r'20200312 true single/'
    tr_path.append(
        root_path + daily_path + r'2.1 500pM brachded 2.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 1/hel147.traces')
    tr_path.append(
        root_path + daily_path + r'2.2 500pM brachded 2.3 + 10nM MF85 (barcode B) + 100nM MF73 (acceptor) round 2/hel150.traces')
    tr_path.append(
        root_path + daily_path + r'2.3 500pM brachded 2.3 + 10nM MF34 (barcode A) + 100nM MF73 (acceptor) round 3/hel153.traces')
elif data_set == 2:
    # branch 1-3
    daily_path = r'20200401 true single molecule branch 1 and 2 long movie/'
    tr_path.append(
        root_path + daily_path + r'1.1 100pM Branch 1 + 10nM MF34 (A) + 100nM MF73 (acceptor) + 500mM NaCl 18k frames/hel1.traces')
    tr_path.append(
        root_path + daily_path + r'1.2 100pM Branch 1 + 10nM MF85 (B) + 100nM MF73 (acceptor) + 500mM NaCl 17k frames/hel2.traces')
elif data_set == 3:
    # branch 2-3
    daily_path = r'20200409 true single molecule branch 1 and 2 TIR-V/'
    tr_path.append(
        root_path + daily_path + r'2.2 50pM branch 2 + 10nM MF85 (barcode B) + 100nM MF73 (acceptor) 500mM NaCl 34k frames/hel4.traces')
    tr_path.append(
        root_path + daily_path + r'2.3 50pM branch 2 + 10nM MF34 (barcode A) + 100nM MF73 (acceptor) 500mM NaCl 25k frames/hel5.traces')
elif data_set == 4:
    # branch 1-3
    daily_path = r'20200423 true single molecule branch 27 and 31 (individual + mixture)/'
    tr_path.append(
        root_path + daily_path + r'3.1 75pM branch 27 + 10nM MF34 (A) and 100nM MF73 (acceptor) 500mM NaCl (15k frames)/hel16.traces')
    tr_path.append(
        root_path + daily_path + r'3.2 75pM branch 27 + 10nM MF85 (B) and 100nM MF73 (acceptor) 500mM NaCl (18k frames)/hel17.traces')
elif data_set == 5:
    # branch 2-3
    daily_path = r'20200423 true single molecule branch 27 and 31 (individual + mixture)/'
    tr_path.append(
        root_path + daily_path + r'5.1 75pM branch 31 + 10nM MF34 (A) and 100nM MF73 (acceptor) 500mM NaCl (15k frames)/hel19.traces')
    tr_path.append(
        root_path + daily_path + r'5.2 75pM branch 31 + 10nM MF85 (B) and 100nM MF73 (acceptor) 500mM NaCl (15k frames)/hel20.traces')
elif data_set == 6:
    # branch mixture
    daily_path = r'20200423 true single molecule branch 27 and 31 (individual + mixture)/'
    tr_path.append(
        root_path + daily_path + r'6.1 75pM 1_1 (branch 27 + branch 31) + 10nM MF34 (A) and 100nM MF73 (acceptor) 500mM NaCl (18.5k frames)/hel21.traces')
    tr_path.append(
        root_path + daily_path + r'6.2 75pM 1_1 (branch 27 + branch 31) + 10nM MF85 (B) and 100nM MF73 (acceptor) 500mM NaCl (18.5k frames)/hel22.traces')

'''   ==========================   '''
'''   ====  Start analysis  ==== '''
while True:
    u_ans = input('Do you want to load previous analysis? [y/n]  ')
    if u_ans == 'y' or u_ans == 'Y':
        dussel_mode = 'load'
        break
    elif u_ans == 'n' or u_ans == 'N':
        dussel_mode = 'new'
        break

''' Rene analysis/load '''
renes = []
# uip = {"e_bin_size": 0.01}
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

    "t_bin_size": 500,
    "e_bin_size": 0.01,
}

for i, c_pth in enumerate(tr_path):
    out_fname = os.path.splitext(c_pth)
    if dussel_mode == 'new':
        # run rene analysis
        # uip["trace_path"] = c_pth
        renes.append(RenePlotKimohist([c_pth], uip))
        # save rene obj
        with open(out_fname[0] + '.rene', 'wb') as rene_file:
            pickle.dump(renes[i], rene_file)
            rene_file.close()
    elif dussel_mode == 'load':
        print(f'Loading rene objects... {os.path.basename(c_pth)}')
        with open(out_fname[0] + '.rene', 'rb') as rene_file:
            tmp_rene = pickle.load(rene_file)
            renes.append(tmp_rene)
n_rene = renes.__len__()


def draw_figures(renes, n_rene, out_path, molecules2show, data_set, draw_as_peak=True):
    ''' make a subfolder for figures '''
    if draw_as_peak:
        print('Saving results with peaks')
        output_fig_path = os.path.join(out_path, f'e{data_set}_figs_peak')
    else:
        print('Saving results with all data points')
        output_fig_path = os.path.join(out_path, f'e{data_set}_figs_dpoint')
    if os.path.exists(output_fig_path):
        shutil.rmtree(output_fig_path)
    os.mkdir(output_fig_path)

    ''' draw hist kymo for all molecules '''
    sys.stdout.write(f'\r   saving kymos for all molecules')
    fig_kymo_hist_allmolecule = Fig_kymo_hist(fig_num=1, n_subplot=n_rene)
    for rene_id, rene_obj in enumerate(renes):
        rene_obj.show_hist_kymo(fig_kymo_hist_allmolecule, rene_id, n_rene,
                                draw_as_peak=draw_as_peak)  # if mid is not given, draw all the molecules
    fig_kymo_hist_allmolecule.ax[0].set_title('All molecules')
    try:
        fig_kymo_hist_allmolecule.fhdl.savefig(os.path.join(output_fig_path, 'all_molecules_kymo.' + graphic_format),
                                               format=graphic_format)
    except:
        pass

    plt.close(fig_kymo_hist_allmolecule.fhdl)

    ''' draw individual molecules '''
    n_mol2show = len(molecules2show) - 1
    for mid in molecules2show:
        sys.stdout.write(f'\r   saving kymos for molecule {mid}/{n_mol2show}')
        fig_kymo_hist_idvmolecule = Fig_kymo_hist(fig_num=mid + 200, n_subplot=n_rene)
        for rene_id, rene_obj in enumerate(renes):
            rene_obj.show_hist_kymo(fig_kymo_hist_idvmolecule, rene_id, n_rene, mid, draw_as_peak=draw_as_peak)
        try:
            fig_kymo_hist_idvmolecule.fhdl.savefig(os.path.join(output_fig_path, f'mol{mid}_kymo.' + graphic_format),
                                                   format=graphic_format)
        except:
            pass
        plt.close(fig_kymo_hist_idvmolecule.fhdl)
    sys.stdout.write('\n')


def draw_traces(renes, n_rene, out_path, molecules2show, data_set):
    output_fig_path = os.path.join(out_path, f'e{data_set}_figs_traces')
    if os.path.exists(output_fig_path):
        shutil.rmtree(output_fig_path)
    os.mkdir(output_fig_path)
    n_mol2show = len(molecules2show) - 1
    for mid in molecules2show:
        sys.stdout.write(f'\r   saving traces for molecule {mid}/{n_mol2show}')
        fig_trace_idvmolecule = Fig_trace(fig_num=mid + 300, n_subplot=n_rene)
        for rene_id, rene_obj in enumerate(renes):
            rene_obj.show_trace(fig_trace_idvmolecule, rene_id, n_rene, mid)
        try:
            fig_trace_idvmolecule.fhdl.savefig(os.path.join(output_fig_path, f'mol{mid}_trace.' + graphic_format),
                                               format=graphic_format)
        except:
            pass
        plt.close(fig_trace_idvmolecule.fhdl)
    sys.stdout.write('\n')


if molecules2show == -1:
    molecules2show = range(len(renes[0].E))
draw_figures(renes, n_rene, os.path.join(root_path + daily_path), molecules2show, data_set, draw_as_peak=True)
draw_figures(renes, n_rene, os.path.join(root_path + daily_path), molecules2show, data_set, draw_as_peak=False)
draw_traces(renes, n_rene, os.path.join(root_path + daily_path), molecules2show, data_set)

print('Job done!')
# plt.show()
