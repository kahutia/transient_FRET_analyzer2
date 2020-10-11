# import matplotlib.pyplot as plt
# import os.path
# import pickle
# from rene_plots import RenePlot
# from rene_plots import Fig_kymo_hist
# from rene_plots import Fig_trace
#
#
# data_set = 0
# molecules2show = range(10)
#
# root_path = r'/Users/sunghyunkim/Desktop/true single/'
# tr_path = list()
# if data_set == 0:
#     # branch 1-3
#     daily_path = r'20200312 true single/test/'
#     tr_path.append(
#         root_path + daily_path + r'1.1 100pM brachded 1.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 1/hel137.traces')
#     tr_path.append(
#         root_path + daily_path + r'1.2 100pM brachded 1.3 + 10nM MF85 (barcode B) + 10nM MF73 (acceptor) round 2/hel141.traces')
#     tr_path.append(
#         root_path + daily_path + r'1.3 100pM brachded 1.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 3/hel144.traces')
# elif data_set == 1:
#     # branch 2-3
#     daily_path = r'20200312 true single/test/'
#     tr_path.append(
#         root_path + daily_path + r'2.1 500pM brachded 2.3 + 10nM MF34 (barcode A) + 10nM MF73 (acceptor) round 1/hel147.traces')
#     tr_path.append(
#         root_path + daily_path + r'2.2 500pM brachded 2.3 + 10nM MF85 (barcode B) + 100nM MF73 (acceptor) round 2/hel150.traces')
#     tr_path.append(
#         root_path + daily_path + r'2.3 500pM brachded 2.3 + 10nM MF34 (barcode A) + 100nM MF73 (acceptor) round 3/hel153.traces')
# elif data_set == 2:
#     # branch 1-3
#     daily_path = r'20200401 true single molecule branch 1 and 2 long movie/'
#     tr_path.append(
#         root_path + daily_path + r'1.1 100pM Branch 1 + 10nM MF34 (A) + 100nM MF73 (acceptor) + 500mM NaCl 18k frames/hel1.traces')
#     tr_path.append(
#         root_path + daily_path + r'1.2 100pM Branch 1 + 10nM MF85 (B) + 100nM MF73 (acceptor) + 500mM NaCl 17k frames/hel2.traces')
# elif data_set == 3:
#     # branch 2-3
#     daily_path = r'20200409 true single molecule branch 1 and 2 TIR-V/'
#     tr_path.append(
#         root_path + daily_path + r'2.2 50pM branch 2 + 10nM MF85 (barcode B) + 100nM MF73 (acceptor) 500mM NaCl 34k frames/hel4.traces')
#     tr_path.append(
#         root_path + daily_path + r'2.3 50pM branch 2 + 10nM MF34 (barcode A) + 100nM MF73 (acceptor) 500mM NaCl 25k frames/hel5.traces')
#
#
#
# '''   ==========================   '''
# '''   ====  Start analysis  ==== '''
#
# ''' read rene object '''
# renes = []
# for c_pth in tr_path:
#     out_fname = os.path.splitext(c_pth)
#     with open(out_fname[0] + '.rene', 'rb') as rene_file:
#         tmp_rene = pickle.load(rene_file)
#         renes.append(tmp_rene)
# n_rene=renes.__len__()
#
# ''' draw hist kymo for all molecules '''
# fig_kymo_hist_allmolecule = Fig_kymo_hist(fig_num=1, n_subplot=n_rene)
# for rene_id, rene_obj in enumerate(renes):
#     rene_obj.show_hist_kymo(fig_kymo_hist_allmolecule, rene_id, n_rene, draw_as_peak=False)  # if mid is not given, draw all the molecules
# fig_kymo_hist_allmolecule.ax[0].set_title('All molecules')
#
# ''' draw individual molecules '''
# fig_kymo_hist_idvmolecule = []
# fig_trace_idvmolecule = []
# for mid in molecules2show:
#     fig_kymo_hist_idvmolecule.append(Fig_kymo_hist(fig_num=mid + 200, n_subplot=n_rene))
#     fig_trace_idvmolecule.append(Fig_trace(fig_num=mid + 300, n_subplot=n_rene))
#     for rene_id, rene_obj in enumerate(renes):
#         rene_obj.show_hist_kymo(fig_kymo_hist_idvmolecule[mid], rene_id, n_rene, mid, draw_as_peak=False)
#         rene_obj.show_trace(fig_trace_idvmolecule[mid], rene_id, n_rene, mid)
#
# plt.show()
