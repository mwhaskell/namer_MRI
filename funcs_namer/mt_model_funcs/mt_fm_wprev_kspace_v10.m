function [ fit_sub, fit_hf, ks_hf ] = mt_fm_wprev_kspace_v10( dM_in, dM_in_indices, Mn, Cfull, km, ...
    km_prior, tse_traj, U , full_msk_pxls, xprev, exp_str, kfilter,pad)


%%                             Precomputations                           %%

% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;

% find shots that had motion params *UPDATED*, only include those in the tse_traj,
% the rest of the kspace data will be populated with km_prior
%%%% NOTE 1/28/19: if this function is only used to find fit_sub, it wouldn't
%%%% actually need to have km_prior passed in, bc it wouldn't care about
%%%% other areas of k-space, it is mostly just as input now for debugging
%%%% and making sure the code is working as it should
dM_in_all2 = zeros(numel(Mn),1);
dM_in_all2(dM_in_indices) = ones(size(dM_in));
dM_in_all_mtx2 = reshape(dM_in_all2, size(Mn));
shots_w_mt = find(sum(dM_in_all_mtx2,2));
tse_traj_sub = tse_traj(shots_w_mt,:);
Ms = Ms(shots_w_mt,:);

%% evaluate forward model
if is_in_parallel
    
    ks = A_v10_wprev_kspace(xprev(full_msk_pxls),U,Cfull,tse_traj_sub,Ms,full_msk_pxls,pad,km_prior);
    
else
    % init pool
    np = feature('numCores');
    p = gcp('nocreate'); % If pool, do not create new one.
    if isempty(p)
        parpool(np);
    end

    ks = A_v10_wprev_kspace(xprev(full_msk_pxls),U,Cfull,tse_traj_sub,Ms,full_msk_pxls,pad,km_prior);
    
end
% weight the kspace data
km_hf = km .* kfilter;
ks_hf = ks .* kfilter;

fit_hf = norm(ks_hf(:)-km_hf(:))/norm(km_hf(:));

%% find fit of just the subset of kspace lines that were updated
updated_slices = tse_traj_sub(:,1);
updated_lines = tse_traj_sub(:,2:end);

km_sub = zeros(size(km));
ks_sub = zeros(size(ks));
for ii = 1:size(km,3)
    if any(ismember(updated_slices,ii))
        shot_indices = find(updated_slices == ii);
        updated_lines_sli = updated_lines(shot_indices);
    else
        updated_lines_sli = [];
    end
    km_sub(updated_lines_sli,:,ii,:) = km(updated_lines_sli,:,ii,:);
    ks_sub(updated_lines_sli,:,ii,:) = ks(updated_lines_sli,:,ii,:);
end

fit_sub = norm(ks_sub(:)-km_sub(:))/norm(km_sub(:));
if isnan(fit_sub)
    fit_sub = fit_hf;
end

if (~isempty(exp_str))
    save(strcat(exp_str,'_fixed_hr_tmp.mat'),'Mn','dM_in','fit')
end

end










