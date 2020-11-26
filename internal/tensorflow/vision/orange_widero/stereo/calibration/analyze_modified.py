# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# +
# Analyze images for which the RT calibration was modified (rotated)
# For large datasets: for loop and minimal loading
# -

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import time
start = time.time()

# + code_folding=[]
# imports
import pickle
import os
import numpy as np
# # %matplotlib notebook
# # %matplotlib notebook
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
from matplotlib.backends.backend_pdf import PdfPages

plt.style.use('dark_background')
# %matplotlib notebook
# %matplotlib notebook

# %load_ext autoreload
# %autoreload 2

# + code_folding=[0]
# %% Functions - only non nans, only above zero, only meaningful - both (flattened)
# non-nan indices:
def nni(a):
    return ~np.isnan(a)

# non-nan array elements:
def nna(a):
    atemp = np.array(a)
    return atemp[nni(a)]

# non-nan other array:
def nno(a, b):
    atemp = np.array(a)
    return atemp[nni(b)]

# non-nan both:
def nnb(a,b):
    atemp = np.array(a)
    inda = nni(a)
    indb = nni(b)
    return a[inda & indb]



def azf(a):
    return a[a > 0]


def mni(a):
    return ((~np.isnan(a)) & (np.array(a) > 0))


def mna(a):
    atemp = np.array(a)
    return atemp[mni(a)]


def mno(a, b):
    atemp = np.array(a)
    return atemp[mni(b)]



# + code_folding=[0]
# %% Function - define rotation label according to index
def intToxyz(a):
    if a == 0:
        s = 'x'
    elif a == 1 or a == 3:
        s = 'y'
    elif a == 2:
        s = 'z'
    elif a ==4:
        s = 'rand'
    return s


# + code_folding=[0]
# %% Function - mean of bins
def binmean(weight_arr, bin_arr, bins, bins_range=[0, 10], keepaxis=None):
    # bins- int or list
    # bin 'weight_arr' according to 'bin_arr'
    if type(bins) is int:
        bin_edges = np.linspace(bins_range[0], bins_range[1], bins + 1)
    else:
        bin_edges = bins
    bin_centers = bin_edges[:-1] + 0.5 * np.diff(bin_edges)
    if keepaxis == None:
        temp_bin = mna(bin_arr)
        temp_weight = nnb(weight_arr, bin_arr)
        n_per_bin, _ = np.histogram(temp_bin, bins=bin_edges)
        sums_binned, _ = np.histogram(temp_bin, bins=bin_edges, weights=temp_weight)

    elif keepaxis is not None:
        temp_weight1 = np.moveaxis(weight_arr, keepaxis, 0)
        naxis = temp_weight1.shape[0]
        temp_weight1 = temp_weight1.reshape((naxis, -1))
        sums_binned = np.zeros((naxis, len(bin_centers)))
        if np.array(bin_arr).shape == np.array(weight_arr).shape:
            temp_bin1 = np.moveaxis(bin_arr, keepaxis, 0).reshape((naxis, -1))
            n_per_bin = np.zeros(sums_binned.shape)
            for j in range(naxis):
                temp_bin2 = nnb(temp_bin1[j, :], temp_weight1[j, :])
                temp_weight2 = nnb(temp_weight1[j, :], temp_bin1[j, :])
                n_per_bin[j, :], _ = np.histogram(temp_bin2, bins=bin_edges)
                sums_binned[j, :], _ = np.histogram(temp_bin2, bins=bin_edges, weights=temp_weight2)
        else:
            temp_bin1 = np.array(bin_arr).flatten()
            temp_bin2 = nna(temp_bin1)
            n_per_bin, _ = np.histogram(temp_bin2, bins=bin_edges)
            for j in range(naxis):
                temp_weight2 = mno(temp_weight1[j, :], temp_bin1)
                sums_binned[j, :], _ = np.histogram(temp_bin2, bins=bin_edges, weights=temp_weight2)


    means_binned = np.where(n_per_bin > 0, sums_binned / n_per_bin, 0)
    return means_binned, n_per_bin, bin_centers, bin_edges


# +
# Prepare load
save_dir = '/mobileye/algo_STEREO3/ohrl/tolerance/'
save_name = 'set'
ind_load = 47
rot_ax = 4 #  axis rotated around. 0,1,2 is x,y,z. 4 is random direction. default=0. Must match the saved file!
sector_name = 'rear' #Change!


pdf_isopen = 0 # '1' for saving a pdf

path = save_dir + save_name + str(ind_load) + '/' + intToxyz(rot_ax) + '-rot' + '/'
pdf = PdfPages(save_dir + 'pdfs/' + 'Set' + str(ind_load) + '_rotation_report.pdf') # for random directions

file_list = [file for file in os.listdir(path)
             if os.path.isfile(os.path.join(path, file))]
nfiles_samps = len(file_list)

# nfile_load = 2000
nfile_load = nfiles_samps
# nfile_load = 20 # for checking


if nfile_load == nfiles_samps:
    ind_files = range(nfile_load)
elif nfile_load < nfiles_samps:
#     ind_files = np.random.choice(range(nfiles_samps), size=nfile_load, replace=False) # not sure there's a real need for randomizing, as the frames 
    ind_files = range(nfile_load)
load_list = [file_list[i] for i in ind_files]

# + code_folding=[]
# Prepare variables
filez_temp = np.load(path + file_list[0])
angles_deg = filez_temp['angles_deg']
n_ang = len(angles_deg)
diff_shape = filez_temp['diff'].shape



# n_binz = 9
n_binz = 18
bin_edgesz = np.concatenate(
    (np.linspace(0, 70, n_binz), [500]))  # 8 bins equally spaced up to 70, then another bin for the rest (large dists.)

bin_centersz = bin_edgesz[:-1] + 0.5* np.diff(bin_edgesz)
z_bins = np.concatenate((bin_centersz[:-1], [85]))

theta0 = 0.07887  # angle ct. 1 pixel for main @ level -1
angs_pix = angles_deg / theta0
# -

print(filez_temp.keys())
print(filez_temp['diff'].shape)

print(filez_temp['rot_ax'])

print(filez_temp['views2mod'])
print(filez_temp['sector'])

# + code_folding=[]
# cam_mod = 'park_rear' # For saving, KEEP TRACK and comment out!

# + code_folding=[]
# Run over files, saves diffs

# diff_arr = np.zeros( (nfile_load,)+diff_shape ) # initialize array for diffs. seems too large, may try w/o
# diff_arr[:] = np.nan
# lidar_imgs = np.zeros( (nfile_load,) + diff_shape[1:3] )
diff_vec = np.zeros((n_ang , 1 )) # first row should be neglected (for initialization)
lidar_vec = np.zeros(1) # first entry should be neglected (for initialization)
file_inds_vec = np.zeros(1) # inds of 'file_list' that are analyzed, first entry should be neglected (for initialization)
SE_binz = np.zeros((n_ang, n_binz))
AE_binz = np.zeros((n_ang, n_binz))
SE_per_frame = np.zeros((n_ang, nfile_load))
AE_per_frame = np.zeros((n_ang, nfile_load))
n_per_frame = np.zeros(nfile_load)
n_per_binz = np.zeros(n_binz)
# print(diff_vec.shape) # for monitoring
# print(lidar_vec.shape) # for monitoring


for c_file, file in enumerate(load_list):
    filez_temp = np.load(path + file)
    diff_temp = filez_temp['diff']
    lidar_temp = filez_temp['lidar_img']
    file_inds_temp = c_file * np.ones(nna(lidar_temp).shape)
    lidar_vec = np.append(lidar_vec, nna(lidar_temp))
    diff_vec = np.concatenate((diff_vec, diff_temp.reshape((n_ang,-1))[:,nni(lidar_temp.flatten())] ) , axis=1   )
    file_inds_vec = np.append( file_inds_vec, file_inds_temp)
#     diff_arr[c_file,:,:,:] = diff_temp # no need for now, takes a lot of RAM
#     lidar_imgs[c_file,:,:] = lidar_temp
    n_per_frame[c_file] = nni(diff_temp[0,:,:]).sum()
    lid_temp2 = nna(lidar_temp)
    n_per_binz_temp, _ = np.histogram( lid_temp2, bins=bin_edgesz  )
    n_per_binz += n_per_binz_temp # n_per_bin: (n_binz,)
    for c_ang in range(n_ang):
        SE_per_frame[c_ang, c_file] = ( nna(diff_temp[c_ang,:,:]) **2 ).sum()
        AE_per_frame[c_ang, c_file] = ( np.abs( nna(diff_temp[c_ang,:,:]) ) ).sum()
        SE_binz_temp, _ = np.histogram( lid_temp2, bins=bin_edgesz, weights=nna(diff_temp[c_ang,:,:])**2 )
        SE_binz[c_ang,:] += SE_binz_temp
        AE_binz_temp, _ = np.histogram( lid_temp2, bins=bin_edgesz, weights=np.abs(nna(diff_temp[c_ang,:,:]) ) )
        AE_binz[c_ang,:] += AE_binz_temp
#     print(diff_vec.shape) # for monitoring
#     print(lidar_vec.shape) # for monitoring
        
SE_all = SE_per_frame.sum(axis=1)
AE_all = AE_per_frame.sum(axis=1)
n_lidar_all = n_per_frame.sum()
# -

print(len([file_list[i] for i in ind_files]))
print(len(ind_files))

# + code_folding=[]
# Complete calculations
RMSE_all = np.sqrt( SE_all/n_lidar_all)
RMSE_frames = np.sqrt( SE_per_frame /  np.where( n_per_frame>0, n_per_frame, np.inf) )
RMSE_binz = np.sqrt( SE_binz /  np.where( n_per_binz>0 , n_per_binz, np.inf) )
E_all = ( AE_all/n_lidar_all)
E_frames = np.sqrt( AE_per_frame /  np.where( n_per_frame>0, n_per_frame, np.inf) )
E_binz =  AE_binz /  np.where( n_per_binz>0 , n_per_binz, np.inf) 

# -

print(E_frames.shape)

# + code_folding=[]
# median calculations
inds_binz = np.digitize(lidar_vec, bin_edgesz )-1
ea_medianz = np.zeros((n_ang, n_binz))
# indsz = np.digitize(  )
for c_z in range(n_binz):
    ea_medianz[:,c_z] = np.median(np.abs(diff_vec[:,inds_binz==c_z ]), axis=1 )
        
ea_median_all = np.median(np.abs(diff_vec), axis=1)
# -

ea_median_all.shape

# + code_folding=[0]
#%% RMSE vs angle for all points
# # %matplotlib notebook
fig1 = plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.plot(angles_deg, RMSE_all , 'o');
plt.xlabel(r"angles [$\degree$]")
plt.ylabel('RMSE for ' + str(nfile_load) + ' frames')
plt.grid(which='both', alpha=0.1)

plt.subplot(1,3,2)
plt.plot(angles_deg, E_all , 'o');
plt.xlabel(r"angles [$\degree$]")
plt.ylabel('L1 error for ' + str(nfile_load) + ' frames')
plt.grid(which='both', alpha=0.1)

plt.subplot(1,3,3)
plt.plot(angles_deg, ea_median_all , 'o');
plt.xlabel(r"angles [$\degree$]")
plt.ylabel('median abs error for ' + str(nfile_load) + ' frames')
plt.grid(which='both', alpha=0.1)

fig1.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note("Average over all frames")
    pdf.savefig(fig1)

# + code_folding=[0]
# %% subplots of RMSE vs angle for different frames
# # %matplotlib notebook
fig4 = plt.figure(figsize=(10,6))
file_inds_plot = np.random.choice(nfile_load, size=min(nfile_load, 9), replace=False)
for file_counter, file_ind in enumerate(file_inds_plot):
    plt.subplot(3, 3, file_counter + 1)
    plt.title(r"file " + str(file_ind) + " #lidar=" +
              str(int(n_per_frame[file_ind])), fontsize=10)
    plt.plot(angs_pix, RMSE_frames[:,file_ind], 'ro', markersize=3)
#     plt.xlabel(r'angle [$\degree$]')
    plt.xlabel(r'angle [pix]')
    plt.ylabel('RMSE')
    plt.grid(which='both', alpha=0.1)
fig4.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note(r"RMSE($\theta$) for several frames")
    pdf.savefig(fig4)

# + code_folding=[]
# %% subplots of abs error vs angle for different frames
# # %matplotlib notebook
fig4b = plt.figure(figsize=(10,6))
file_inds_plot = np.random.choice(nfile_load, size=min(nfile_load, 9), replace=False)
for file_counter, file_ind in enumerate(file_inds_plot):
    plt.subplot(3, 3, file_counter + 1)
    plt.title(r"file " + str(file_ind) + " #lidar=" +
              str(int(n_per_frame[file_ind])), fontsize=10)
    plt.plot(angs_pix, E_frames[:,file_ind], 'ro', markersize=3)
#     plt.xlabel(r'angle [$\degree$]')
    plt.xlabel(r'angle [pix]')
    plt.ylabel('abs err')
    plt.grid(which='both', alpha=0.1)
fig4b.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note(r"error($\theta$) for several frames")
    pdf.savefig(fig4b)

# + code_folding=[0]
# %% Plot binned RMSE vs theta for different z
# # %matplotlib notebook
fig2 = plt.figure(figsize=(10,16))
for bin_countz, binc in enumerate(bin_centersz):
    plt.subplot(n_binz/3, 3, bin_countz + 1)
    plt.plot(angs_pix, RMSE_binz[:, bin_countz], 'mo', markersize=2)
    plt.xlabel(r"angle [$\degree$]")
    plt.xlabel(r"angle [pix]")
    plt.ylabel('RMSE')
    plt.title('z=[%.1f %.1f], num=%d' % (bin_edgesz[bin_countz], bin_edgesz[bin_countz+1],
                        n_per_binz[bin_countz]), fontsize=8)
    plt.grid(which='both', alpha=0.1)
fig2.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note(r"RMSE($\theta$) for several z")
    pdf.savefig(fig2)


# + code_folding=[0]
# %% Plot binned abs error vs theta for different z
# # %matplotlib notebook
fig2b = plt.figure(figsize=(8,16))
for bin_countz, binc in enumerate(bin_centersz):
    plt.subplot(n_binz/3, 3, bin_countz + 1)
    plt.plot(angs_pix, E_binz[:, bin_countz], 'mo', markersize=2)
#     plt.xlabel(r"angle [$\degree$]")
    plt.xlabel(r"angle pix$]")
    plt.ylabel('abs err')
    plt.title('z=[%.1f %.1f], num=%d' % (bin_edgesz[bin_countz], bin_edgesz[bin_countz+1],
                        n_per_binz[bin_countz]), fontsize=8)
    plt.grid(which='both', alpha=0.1)
fig2b.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note(r"error($\theta$) for several z")
    pdf.savefig(fig2b)

# + code_folding=[0]
# %% Plot median binned abs error vs theta for different z
# # %matplotlib notebook
fig2c = plt.figure(figsize=(8,16))
for bin_countz, binc in enumerate(bin_centersz):
    plt.subplot(n_binz/3, 3, bin_countz + 1)
    plt.plot(angs_pix, ea_medianz[:, bin_countz], 'mo', markersize=2)
    plt.xlabel(r"angle [pix]")
    plt.ylabel('median err')
    plt.title('z=[%.1f %.1f], num=%d' % (bin_edgesz[bin_countz], bin_edgesz[bin_countz+1],
                        n_per_binz[bin_countz]), fontsize=8)
    plt.grid(which='both', alpha=0.1)
fig2c.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note(r"median error($\theta$) for several z")
    pdf.savefig(fig2c)

# + code_folding=[]
# Calculate error at theta0 (ct. 1 pixel@main@-1)
ind1 = np.argmin(np.abs(angles_deg - theta0))
ind2 = np.argmin(np.abs(angles_deg - 2*theta0))
eb_binsz = RMSE_binz[0,:]
e1_binsz = np.sqrt(  ( (RMSE_binz[ind1,:]**2) - ( RMSE_binz[0,:]**2) )  * ( RMSE_binz[ind1,:] >  RMSE_binz[0,:]   ) )
eab_binsz = E_binz[0,:]
ea1_binsz =  ( E_binz[ind1,:] -  E_binz[0,:] )  * ( E_binz[ind1,:] >  E_binz[0,:]   ) 
ea2_binsz =  ( E_binz[ind2,:] -  E_binz[0,:] )  * ( E_binz[ind2,:] >  E_binz[0,:]   ) 


# + code_folding=[6]
# functions for fitting to a line (order one polynomial) w/o a constant/offset
def polyz1(x, p):
    # pb = np.broadcast_to(p,x.shape)
    y = p*x
    return y

def fit_polyz1(x,y):
    p, _= curve_fit(polyz1,x,y)
    return p



# -

ea_medianz

# + code_folding=[]
# linear fit of abs error and median vs z
ea_plin1 = fit_polyz1(z_bins, ea1_binsz)
ea_plin2 = fit_polyz1(z_bins, ea2_binsz)
med_plin1 = fit_polyz1(z_bins, ea_medianz[ind1,:]-ea_medianz[0,:])
med_plin2 = fit_polyz1(z_bins, ea_medianz[ind2,:]-ea_medianz[0,:])
print("%.2e" % ea_plin1)
print("%.2e" % ea_plin2)
print("%.2e" % med_plin1)
print("%.2e" % med_plin2)

# + code_folding=[0]
# %% Plot of bare and added (e1) errors (RMSE) vs distance z
# # %matplotlib notebook
fig6 = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(z_bins, e1_binsz, 'ro', markersize=3)
# plt.plot(z_bins, polyz1(z_bins, ppolz1),'y')
# plt.plot(z_bins, polyz2(z_bins, ppolz2[0], ppolz2[1]),'c')
plt.xlabel('z [m]')
plt.ylabel(r'$e_1$ error at angle of 1 pixel')
ylim1 = plt.gca().get_ylim()
# plt.text(5, 0.7*ylim1[1], 'linear slope:\n %.1e' % (ppolz1))
# plt.gca().tick_params(axis='x', which='minor', bottom=False)
plt.grid(which='both', alpha =0.1)


plt.subplot(2, 1, 2)
plt.plot(z_bins, eb_binsz, 'ro', markersize=3)
plt.xlabel('z [m]')
plt.ylabel(r'"bare" error')
plt.grid(which='both', alpha=0.1)

fig6.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note("Plot of curvature a and offset b vs distance z")
    pdf.savefig(fig6)

# + code_folding=[]
# %% Plot of bare and added (e1) errors (abs) vs distance z
# # %matplotlib notebook
fig6b = plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1)
plt.plot(z_bins, polyz1(z_bins, med_plin1),'y')
plt.plot(z_bins, polyz1(z_bins, med_plin2),'c')
plt.plot(z_bins, ea1_binsz, 'ro', markersize=3)
plt.plot(z_bins, ea2_binsz, 'go', markersize=3)
plt.xlabel('z [m]')
plt.ylabel(r'added abs error at angle of 1 pixel')
ylim1 = plt.gca().get_ylim()
plt.text(5, 0.7*ylim1[1], 'linear slopes:\n %.2e, %.2e' % (ea_plin1, ea_plin2))
plt.text(45, ea_plin2*45*1.5, r'2 pix', color='g' )
plt.text(70, ea_plin1*70*1.35, r'1 pix', color='r' )
# plt.gca().tick_params(axis='x', which='minor', bottom=False)
plt.grid(which='both', alpha =0.1)
# plt.set_ylim()

plt.subplot(2, 1, 2)
plt.plot(z_bins, eab_binsz , 'ro', markersize=3)
plt.xlabel('z [m]')
plt.ylabel(r'"bare" ab error')
plt.grid(which='both', alpha=0.1)
# 
# fig6b.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note("Plot of curvature a and offset b vs distance z")
    pdf.savefig(fig6b)

# + code_folding=[]
# %% Plot of bare and added errors (medians) vs distance z
# # %matplotlib notebook
fig6c = plt.figure(figsize=(10,8))

plt.subplot(2, 1, 1)
plt.plot(z_bins, polyz1(z_bins, med_plin1),'y')
plt.plot(z_bins, polyz1(z_bins, med_plin2),'c')
plt.plot(z_bins, ea_medianz[ind1,:] - ea_medianz[0,:], 'ro', markersize=3)
plt.plot(z_bins, ea_medianz[ind2,:] - ea_medianz[0,:], 'go', markersize=3)
plt.xlabel('z [m]')
plt.ylabel(r'added median at angle of 1 pixel')
ylim1 = plt.gca().get_ylim()
plt.text(5, 0.7*ylim1[1], 'linear slopes:\n %.2e, %.2e' % (med_plin1, med_plin2))
plt.text(45, med_plin2*45*1.5, r'2 pix', color='g' )
plt.text(70, med_plin1*70*1.35, r'1 pix', color='r' )
# plt.gca().tick_params(axis='x', which='minor', bottom=False)
plt.grid(which='both', alpha =0.1)
# plt.set_ylim()

plt.subplot(2, 1, 2)
plt.plot(z_bins, ea_medianz[0,:] , 'ro', markersize=3)
plt.xlabel('z [m]')
plt.ylabel(r'median "bare" ab error')
plt.grid(which='both', alpha=0.1)
# 
fig6c.tight_layout()
plt.show()
if pdf_isopen:
    pdf.attach_note("Plot of curvature a and offset b vs distance z")
    pdf.savefig(fig6c)

# +
#%% Plot of median error (norm. by bare) vs angle[pix] at 40m
ind_binz40 = np.argmin(np.abs(z_bins-40))
# # %matplotlib notebook
fig7 = plt.figure(figsize=(8,6))
norm_added_med = ( ea_medianz[:,ind_binz40] - ea_medianz[0, ind_binz40])  / ea_medianz[0, ind_binz40]
plt.plot( angs_pix, norm_added_med, 'or' )
plt.xlabel('angle [pix @ main level -1]', fontsize=18)
plt.ylabel('median absolute error (norm.)'  ,fontsize=18)
plt.title('z = %d [m]' % z_bins[ind_binz40] , fontsize=18)
plt.grid(which='both', alpha=0.2)
plt.show()
# plt.savefig(save_dir + 'results/'+ sector_name + '/' + cam_mod + '_median_40m.pdf') # for saving some "final" result 
# plt.savefig(save_dir + 'results/front/FCR_median_40m.pdf') # for saving some "final" result 
# plt.savefig(save_dir + 'results/front/FCL_median_40m.pdf') # for saving some "final" result 
# plt.savefig(save_dir + 'results/front/park_median_40m.pdf') # for saving some "final" result 

if pdf_isopen:
    pdf.savefig(fig7)

# z_bins[ind_binz40]

# -

save_dir

print(ea_medianz.shape)
print(ea_medianz[0,ind_binz40])

save_dir

z_bins

# +
# np.savez(save_dir+'results/'+ sector_name + '/' + cam_mod + '.npz' ,ea_medianz=ea_medianz,
#          angles_deg=angles_deg, angs_pix=angs_pix,
#         diff_vec=diff_vec, lidar_vec=lidar_vec, file_inds_vec=file_inds_vec, file_list=file_list, z_bins=z_bins)
# os.system("printf '\a'") 
# os.system("printf '\a'") 

# + code_folding=[1]
end = time.time()
if pdf_isopen:
    pdf.close()
    pdf_isopen=0


print("Finished after %.2f" %(end-start))

# -

print( (end-start)/60.)

print(diff_vec.shape)
# print(diff.shape)

# + code_folding=[0]
# Show histogram
# %matplotlib notebook
plt.figure()
axtemp = plt.hist(diff_vec[ind1,:].flatten() , bins=650, range=(-2,2))
# plt.gca().set_xlim([-2,2])
plt.show()

# + code_folding=[0]
# A few statistics
bound =0.1
# indstat= ind2
indstat=-1
# indstat=0
print((np.abs(diff_vec[indstat,:])>=bound).sum())
print(diff_vec[indstat,:].size)
print((np.abs(diff_vec[indstat,:])>=bound).sum() *1. / diff_vec[indstat,:].size)
print(diff_vec[indstat,:].std() )


# + code_folding=[0]
# Function for the Cauchy distribution
# As in wikipedia https://en.wikipedia.org/wiki/Cauchy_distribution
def cauchy(x,a, x0, gamma ):
    y = 1./ ( np.pi * gamma * ( 1 +  ( (x-x0)/ gamma ) **2  ))
    return y

def logcauchy(x,a, x0, gamma ):
    y = np.log(cauchy(x,a, x0, gamma ))
    return y



# + code_folding=[]
# fit histogram to a Cauchy distribution
from scipy.optimize import curve_fit
# ind_hist = ind1
# ind_hist = -1
ind_hist = 0
diff_hist, diff_bin_edges = np.histogram(diff_vec[ind_hist,:], bins=12001 , density=True)
diff_bin_centers = diff_bin_edges[:-1] + 0.5 * np.diff(diff_bin_edges)

popt_c, pconv_c = curve_fit(cauchy,diff_bin_centers,diff_hist, p0=(1, 0,0.02))
# popt_c, pconv_c = curve_fit(logcauchy,diff_bin_centers[diff_hist>0],np.log(diff_hist[diff_hist>0]), p0=(1, 0,0.02))

fig = plt.figure()
plt.plot(diff_bin_centers, diff_hist)
plt.plot(diff_bin_centers, cauchy(diff_bin_centers, popt_c[0], popt_c[1], popt_c[2]), '--r')
plt.xlabel('diff')
plt.ylabel('probability')
plt.gca().set_xlim((-3,3))
plt.gca().set_yscale('log')
plt.show()
# -

for i in range(6):
    time.sleep(0.5)
    os.system("printf '\a'") 
