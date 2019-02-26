% namer_recon.m

% Example code for the method described in:

% Haskell et al. 2019, "Network Accelerated Motion 
% Estimation and Reduction (NAMER): Convolutional neural network guided 
% retrospective motion correction using a separable motion model"

% This script performs the separable cost function version of the NAMER 
% method (Eqn 3 in Haskell et al. 2019), and corresponds to the result
% shown in the bottom left of Figure 4-B in the paper.


%% Initialize script, set filenames

NAMER_path = [pwd,'/'];
addpath(genpath('./funcs_namer'))
addpath('./namer_data')

exp_name = '_namer_example';
niters = 20;
gpu_str = '0';
data_fn = 'namer_ex_data.mat';
motion_fn = 'subject_mt_example.mat';

% cnn name and patch variables
cnn_model_name = 'namer_trained_model.h5';
cnn_tmp_path = NAMER_path;
patch_params.patch_sz = 51;
patch_params.patch_stsz = 8;
create_patches = true;
patches_full_fn = [NAMER_path, datestr(now,'yy-mm-dd'),'_namer_patches_tmp'];

%% Load channel data, find relevant parameters

% load k-space, sensitivity maps, shot trajectory, undersampling matrix,
% kspace filter (all ones in this example), and mask of non-zero pixels
load(data_fn)
[nlin,ncol,nsli_0,ncha] = size(kdata);
shots_per_slice = size(tse_traj,1); 
total_shots = shots_per_slice * nsli_0;

% pad data as needed 
pad = 0; % (non zero for volumes with through-slice motion correction)
nsli_p = nsli_0 + 2*pad;
kdata = cat(3, zeros(nlin,ncol,pad,ncha),kdata,zeros(nlin,ncol,pad,ncha));
sens = cat(3, repmat(sens(:,:,1,:),[1 1 pad]),sens,repmat(sens(:,:,end,:),[1 1 pad]));
k_in = kdata;

% load motion trajectory. This matrix is (number of shots x 6), for the 6
% rigid body motion parameters
load(motion_fn)

% create dM_in_indicies. This variable determines which motion parameters
% will be corrected. Here we only do in-plane correction, so columns
% corresponding to through plane translation and rotation are set to zero
dM_in_matrix = ones(size(mt_traj));
dM_in_matrix(1,:) = 0;
dM_in_matrix(:,3:5) = 0;
dM_in_indices = find(dM_in_matrix);

% zero motion variables (used for initial reconstructions)
Mz = zeros(shots_per_slice,6);
dM_z = zeros(numel(dM_in_indices),1);

% create experiment strings
exp_str = strcat(datestr(now,'yyyy-mm-dd'),exp_name,'_');
exp_path = exp_str; exp_path(end) = '/';
while exist(exp_path,'dir')
    exp_path = strcat(exp_path(1:end-1),'i/');
    exp_str = strcat(exp_str(1:end-1),'i_');
end
mkdir(exp_path);
full_path_exp_str = strcat(NAMER_path,'/',exp_path,exp_str);


%% Find no motion image and motion corrupted image

% find original image before motion corruption
[ fit_org, xorg, ~] = mt_fit_fcn_v10( dM_z, dM_in_indices, Mz, sens, kdata, ...
    tse_traj, U , nz_pxls , nz_pxls, [], [], kfilt, pad);

% simulate R = 1 kspace for simulating the effects of motion (raw data is
% R = 2 which won't work with the addmotion function)
sim_R1_coil_data = fft2c(sens.*repmat(xorg,1,1,1,ncha));
k_R1 = zeros(size(kdata));
U_nonzero = find(U); U_zero = find(U-1);
k_R1(U_nonzero) = kdata(U_nonzero);
k_R1(U_zero) = sim_R1_coil_data(U_zero);

% add motion affects
km = zeros(size(k_in));
for jj = 1:ncha,km(:,:,:,jj) = addmotionTSE(k_R1(:,:,:,jj),mt_traj,tse_traj,pad);end

% reconstruct simulated motion corrupted data (assuming no motion), and
% reconstruct assuming a perfect estimate of the motion
[ fit_corrupted, x_corrupted, ~] = mt_fit_fcn_v10( dM_z, dM_in_indices, ...
    Mz, sens, km, tse_traj, U , nz_pxls , nz_pxls, [], [], kfilt, pad);
[ fit_w_true_mt, x_w_true_mt, ~] = mt_fit_fcn_v10( dM_z, dM_in_indices, ...
    mt_traj, sens, km, tse_traj, U , nz_pxls , nz_pxls, [], [], kfilt, pad);

%% Loop through NAMER iterations

% init variables for NAMER iterations
x_current = x_corrupted;
M_current = Mz;
fits_post_cnn = zeros(niters,1);
fits_post_mtmin = zeros(niters,1);
fits_full_solve = zeros(niters,1);

disp('  '); disp(strcat(exp_str,'1-',num2str(niters))); disp('  ')
for ii = 1:niters
    
    disp('  '); disp(strcat(exp_str,num2str(ii))); disp('  ');
    diary(strcat(exp_path,exp_str,num2str(ii)))
    
    %% %%%%%      NAMER step 1: apply CNN   %%%%%%%%%%%%%%%%%%%%%
    
    % reformat data to orientation used with CNN
    x_in = permute(x_current(10:end-9,end:-1:1), [2 1]);
    
    % remove artifacts from image by applying CNN
    [x_cnn] = run_cnn(x_in, create_patches, patches_full_fn, ...
        patch_params, cnn_model_name, cnn_tmp_path, gpu_str);
    
    % reformat back to (PE, RO, SLI, CHA) orientation
    x_cnn2 = permute(x_cnn,[2 1]);
    x_cnn3 = cat(1,zeros(9,448),x_cnn2(:,end:-1:1),zeros(9,448));
    
    % find cmplx scale between ML image and initial motion corrupted image
    cf = fminsearch(@(x) nrm_err(x, x_cnn3, x_corrupted), [1;0]);
    x_cnn_final = x_cnn3 * (cf(1) + 1i * cf(2));
    
    % find data consistency fit after application of CNN
    fit_cnn = mt_fm_v10( dM_z, dM_in_indices, M_current, sens, km, ...
        tse_traj, U ,  nz_pxls, x_cnn_final(:), [], kfilt, pad);
    fits_post_cnn(ii) = fit_cnn;
    
    %% %%%%%      NAMER step 2: motion optimization   %%%%%%%%%%%%%%%%%%%%%
    
    % set convergence options
    mt_opt_options = optimoptions(@fminunc, 'Algorithm','quasi-newton',...
        'Display','off','SpecifyObjectiveGradient',false,...
        'OptimalityTolerance',1e-3,'MaxIterations',10);
    
    % create vars for parallel loop and then run separate minimizations for
    % each shot
    dM_update_vals = cell(total_shots,1);
    x_cnn_rep = repmat(x_cnn_final(:),1,total_shots);
    parfor jj = 1:total_shots
        
        % set indicies of motion to be optimized for shot jj
        dM_in_matrix_tmp = zeros(size(Mz));
        dM_in_matrix_tmp(jj,[1,2,6]) = 1;
        dM_in_indices_tmp = find(dM_in_matrix_tmp);
        
        % run motion optimization for shot jj
        [mt_jst_tmp] = fminunc(@(x)mt_fm_v10( x, dM_in_indices_tmp,...
            M_current, sens, km, tse_traj, U ,  nz_pxls, x_cnn_rep(:,jj), ...
            [], kfilt, pad), zeros(3,1), mt_opt_options);
        
        % save optimization output
        dM_update_vals{jj} = mt_jst_tmp;
        
        % display progress
        disp(['Done optimizing shot: ',num2str(jj)])
    end
    
    % collect motion from each shot into the current estimate of the motion
    for jj = 1:total_shots
        M_current(jj,[1,2,6]) = M_current(jj,[1,2,6]) + dM_update_vals{jj}';
    end
    
    % find fit with new motion (but still CNN image)
    [fit_mt_min] = mt_fit_fcn_v10( dM_z, dM_in_indices, M_current, sens, km, ...
        tse_traj, U, [], nz_pxls, x_cnn_final(:), [], kfilt, pad);
    fits_post_mtmin(ii) = fit_mt_min;
    
    disp('Motion optimization completed. Resolving for volume...')
    
    %% %%%%%    NAMER step 3: full volume solve   %%%%%%%%%%%%%%%%%%%%%
    
    % resolve for the full volume with the new motion estimate
    [ fit_full_solve, x_current, pcg_out] = mt_fit_fcn_v10( dM_z, ...
        dM_in_indices, M_current, sens, km, ...
        tse_traj, U , nz_pxls , nz_pxls, [], [], kfilt, pad);
    fits_full_solve(ii) = fit_full_solve;
    
    save(strcat(full_path_exp_str,num2str(ii),'.mat'),'fit_cnn',...
        'x_cnn_final','fit_mt_min', 'fit_full_solve','x_current',...
        'M_current')
    
    
end

%% plot image results and compare to original
sc = .5*max(abs(xorg(:)));
mosaic(permute(xorg(:,end:-1:1),[2,1]),1,1,1,'x orginal',[0 sc]);
mosaic(permute(x_corrupted(:,end:-1:1),[2,1]),1,1,2,'x motion corrupted',[0 sc]);
mosaic(permute(x_w_true_mt(:,end:-1:1),[2,1]),1,1,3,'x with true motion',[0 sc]);
mosaic(permute(x_current(:,end:-1:1),[2 1]),1,1,4,'x - NAMER output',[0 sc])

%% plot motion results and convergence
figure(5); plot(mt_traj(:,[1,2,6])); hold on; plot(M_current(:,[1,2,6]),'k--')
legend('PE mt true','RO mt true','rotation true',...
    'PE mt NAMER','RO mt NAMER','rotation NAMER')
figure(6); plot(0:20,[fit_corrupted;fits_full_solve])
title('data consistency convergence')


%% save final workspace
save(strcat(full_path_exp_str,'end_wrksp.mat'))



