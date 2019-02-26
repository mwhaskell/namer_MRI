function [ fit_hf, x, pcg_out, k_fm] = mt_fit_fcn_v10( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U , tar_pxls , full_msk_pxls, xprev, exp_str, kfilter,pad)


%%                             Precomputations                           %%

%%% Currently hardcoded values
iters = 20;
lambda = 0;

[nlin, ncol, nsli, ~] = size(U);
fixed_pxls = setdiff(full_msk_pxls,tar_pxls);

% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;


%% call pcg

xs_v_f = xprev(fixed_pxls);
Afxf = A_v10(xs_v_f,U,Cfull,tse_traj,Ms,fixed_pxls,pad);
AtsAfxf = Astar_v10(Afxf,U,Cfull,tse_traj,Ms,tar_pxls,pad);
RHS = Astar_v10(km,U,Cfull,tse_traj,Ms,tar_pxls,pad) - AtsAfxf;

if (~isempty(xprev))
    [xs_v_t, f, rr, it] = pcg(@(x)...
        LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,pad), RHS, 1e-3, iters, [], [],...
        reshape(xprev(tar_pxls),numel(tar_pxls),1));
else
    [xs_v_t, f, rr, it] = pcg(@(x)...
        LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,pad), RHS, 1e-3, iters);
end
pcg_out = [f, rr, it];

%% Evaluate Forward Model                             

xs_v_vol = zeros(nlin,ncol,nsli);
xs_v_vol(fixed_pxls) = xs_v_f; xs_v_vol(tar_pxls) = xs_v_t;
xs_v_all = xs_v_vol(full_msk_pxls);

pxl_per_sli = numel(full_msk_pxls)/nsli;
xs_v = zeros(numel(full_msk_pxls),1);
xs_v(pad*pxl_per_sli+1:end-pad*pxl_per_sli) = xs_v_all(pad*pxl_per_sli+1:end-pad*pxl_per_sli);

ks = A_v10(xs_v,U,Cfull,tse_traj,Ms,full_msk_pxls,pad);

k_fm = ks;

% weight the kspace data
km_hf = km .* kfilter;
ks_hf = ks .* kfilter;

fit_hf = norm(ks_hf(:)-km_hf(:))/norm(km_hf(:));

if (~isempty(exp_str))
    save(strcat(exp_str,'_tmp.mat'),'Mn','dM_in','fit_hr')
end

%% put x back in full image matrix
x = zeros(nlin,ncol,nsli);
x(full_msk_pxls) = xs_v;


%% update tamer_vars
global tamer_vars
if isfield(tamer_vars,'track_opt')
    if tamer_vars.track_opt == true
        tamer_vars.nobjfnc_calls = tamer_vars.nobjfnc_calls + 1;
        tamer_vars.pcg_steps = [tamer_vars.pcg_steps, it];
        tamer_vars.ntotal_pcg_steps = tamer_vars.ntotal_pcg_steps + it;
        tamer_vars.fit_vec = [tamer_vars.fit_vec, fit_hf];
    end
end
end


%%           LHS function                                        %%%%%%%%%%
function [output] = LHS_v10(x,U,Cfull,tse_traj,Ms,lambda,nz_pxls,pad)

AsAx = AsA_v10(x,U,Cfull,tse_traj,Ms,nz_pxls,pad);
output = AsAx + lambda;

end







