function [imdata] = Astar_v10(k,U,Cfull,tse_traj,Ms,nz_pxls_in,pad)

%% init pool if needed
if ~is_in_parallel
    % init pool
    np = feature('numCores');
    p = gcp('nocreate'); % If pool, do not create new one.
    if isempty(p)
        parpool(np);
    end
end


%% create cell to hold new tse_traj_cell variable.
% This will combine all of
% the lines from a given slice that had the same motion into one shot (i.e.
% if tse_traj was originally 17x11 but the first two shots had the same
% motion parameters (and the rest were unique) the row element of the cell
% would contain a 22x1 vector with the lines from shots 1 and 2. The, the
% next 15 elements of the cell would contain 11x1 vectors with the lines
% from the rest of the shots.
tse_traj_cell = tse_traj_to_compact_cell(tse_traj,Ms);

tls = size(tse_traj_cell,1);

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_in) = 1;
sli_pxls_in = find(nz_im(:,:,round(end/2)));

%% begin reverse model

% prep variables for parellelization
sli_traj = cell2mat(tse_traj_cell(:,1));
shot_traj = tse_traj_cell(:,2);
Ms_unique = cell2mat(tse_traj_cell(:,3));
dz_v = Ms_unique(:,3);
yaw_v = Ms_unique(:,4); pitch_v = Ms_unique(:,5); roll_v = Ms_unique(:,6);

%%% Uu* operator
% reorganizes data into format based on shots
Uusk = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    tmp_sli = sli_traj(t) + pad;
    tmp_tse_traj = cell2mat(shot_traj(t));
    Uusk(tmp_tse_traj,:,t,:) = k(tmp_tse_traj,:,tmp_sli,:);
end

%%% Fen* operator
FensUusk = fftshift(fftshift( ifft2(ifftshift(ifftshift(Uusk,1),2)) ,1),2);

%%% C* operator
CsFensUusk = zeros(nlin,ncol,tls);
for t = 1:tls
    tmp_sli = sli_traj(t) + pad;
    CsFensUusk(:,:,t) = sum(conj(Cfull(:,:,tmp_sli,:)) .* FensUusk(:,:,t,:),4);
end

%%% Fxyin* operator
FxyinsCsFensUssk=fftshift(fftshift(fft2(ifftshift(ifftshift(CsFensUusk,1),2)),1),2);

%%% Mxy* operator
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);
MxysFxyinsCsFensUssk = zeros(nlin,ncol,tls);
for t = 1:tls
    dx = Ms_unique(t,1);
    dy = Ms_unique(t,2);
    mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
    Mxy_sli = exp(1i * sum(kspace_2d.*mmtx_sli,3) );
    MxysFxyinsCsFensUssk(:,:,t) = Mxy_sli .* FxyinsCsFensUssk(:,:,t);
end

%%% Fxy* operator
FxysMxysFxyinsCsFensUssk = fftshift(fftshift(  ifft2(...
    ifftshift(ifftshift(  MxysFxyinsCsFensUssk, 1), 2)), 1), 2);

% RMk stands for "reverse model k", which will be a sum of the effects of
% each kspace shot on the image
RMk_all = zeros(numel(sli_pxls_in)*nsli,1,tls);

%%% NOTE!!! The "for" loop and "parfor" loop should have identical contents
if is_in_parallel
    for t = 1:tls
        
        tmp_sli = sli_traj(t) + pad;
        
        % create new nx_pxls and sli_pxls
        yaw    = yaw_v(t);
        pitch  = pitch_v(t);
        roll   = roll_v(t);
        
        temp_vol2 = zeros(nlin,ncol,nsli);
        temp_vol2(nz_pxls_in) = 1;
        temp_vol2 = rot3d_v10(temp_vol2,yaw,pitch,roll);
        nz_pxls_pre_rot = find(temp_vol2);
        nz_pxls_all = union(nz_pxls_pre_rot,nz_pxls_in); % not sure if we need nz_pxls_in here
        
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_all) = 1;
        sli_pxls_all = find(sum(nz_im,3));
        
        %%% Uss* operator
        tmp_imsp = zeros(nlin,ncol,nsli);
        tmp_imsp(:,:,tmp_sli) = FxysMxysFxyinsCsFensUssk(:,:,t);
        
        %%% Fz-* operator
        tmp_imsp2 = reshape(tmp_imsp,nlin*ncol,nsli);
        tmp_imsp2 = tmp_imsp2(sli_pxls_all,:);
        obj_xy_kz = fftshift(fft(ifftshift(tmp_imsp2,2), nsli, 2),2);
        
        %%% Mz* operator
        dz = dz_v(t);
        kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
        p_ph = exp(1i * kp_vec     * dz).';        % par phase
        Mz_obj_xy_kz = permute(repmat(p_ph,1,numel(sli_pxls_all)),[ 2 1]) .* obj_xy_kz;
        
        %%% Fz* operator
        obj_xyz = fftshift(ifft(ifftshift(Mz_obj_xy_kz,2), nsli, 2) ,2);
        
        %%% R* operator
        temp_vol = zeros(nlin*ncol,nsli);
        temp_vol(sli_pxls_all, :) = obj_xyz;
        temp_vol = reshape(temp_vol, nlin, ncol, nsli);
        
        inv_rot_vol = rot3d_v10(temp_vol,-1*yaw,-1*pitch,-1*roll);
        
        RMk_all(:,t) = reshape(inv_rot_vol(nz_pxls_in),numel(nz_pxls_in),1);
    end
else
    parfor t = 1:tls
        
        tmp_sli = sli_traj(t) + pad;
        
        % create new nx_pxls and sli_pxls
        yaw    = yaw_v(t);
        pitch  = pitch_v(t);
        roll   = roll_v(t);
        
        temp_vol2 = zeros(nlin,ncol,nsli);
        temp_vol2(nz_pxls_in) = 1;
        temp_vol2 = rot3d_v10(temp_vol2,yaw,pitch,roll);
        nz_pxls_pre_rot = find(temp_vol2);
        nz_pxls_all = union(nz_pxls_pre_rot,nz_pxls_in); % not sure if we need nz_pxls_in here
        
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_all) = 1;
        sli_pxls_all = find(sum(nz_im,3));
        
        %%% Uss* operator
        tmp_imsp = zeros(nlin,ncol,nsli);
        tmp_imsp(:,:,tmp_sli) = FxysMxysFxyinsCsFensUssk(:,:,t);
        
        %%% Fz-* operator
        tmp_imsp2 = reshape(tmp_imsp,nlin*ncol,nsli);
        tmp_imsp2 = tmp_imsp2(sli_pxls_all,:);
        obj_xy_kz = fftshift(fft(ifftshift(tmp_imsp2,2), nsli, 2),2);
        
        %%% Mz* operator
        dz = dz_v(t);
        kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
        p_ph = exp(1i * kp_vec     * dz).';        % par phase
        Mz_obj_xy_kz = permute(repmat(p_ph,1,numel(sli_pxls_all)),[ 2 1]) .* obj_xy_kz;
        
        %%% Fz* operator
        obj_xyz = fftshift(ifft(ifftshift(Mz_obj_xy_kz,2), nsli, 2) ,2);
        
        %%% R* operator
        temp_vol = zeros(nlin*ncol,nsli);
        temp_vol(sli_pxls_all, :) = obj_xyz;
        temp_vol = reshape(temp_vol, nlin, ncol, nsli);
        
        inv_rot_vol = rot3d_v10(temp_vol,-1*yaw,-1*pitch,-1*roll);
        
        RMk_all(:,t) = reshape(inv_rot_vol(nz_pxls_in),numel(nz_pxls_in),1);
    end %%% end parallel loop
    
end %%% end shot loop


RMk = sum(RMk_all,3);
imdata = RMk;

end
