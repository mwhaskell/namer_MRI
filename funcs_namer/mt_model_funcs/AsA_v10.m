function [xprj] = AsA_v10(x,U,Cfull,tse_traj,Ms,nz_pxls_in,pad)

%% init pool if needed
if ~is_in_parallel
    % init pool
    np = feature('numCores');
    p = gcp('nocreate'); % If pool, do not create new one.
    if isempty(p)
        parpool(np);
    end
end

%% precomputations

% alter tse_traj so there are no repeating motion states for any slice
tse_traj_cell = tse_traj_to_compact_cell(tse_traj,Ms);
tls = size(tse_traj_cell,1);
xprj_all = zeros(numel(x),tls);

% precomputations
[nlin, ncol, nsli, ncha] = size(U);

% prep variables for parellelization
sli_traj = cell2mat(tse_traj_cell(:,1));
shot_traj = tse_traj_cell(:,2);
Ms_unique = cell2mat(tse_traj_cell(:,3));
dx_v = Ms_unique(:,1); dy_v = Ms_unique(:,2); dz_v = Ms_unique(:,3);
yaw_v = Ms_unique(:,4); pitch_v = Ms_unique(:,5); roll_v = Ms_unique(:,6);

% create vars for adding motion

kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);

Cfull2 = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    tmp_sli = sli_traj(t) + pad;
    Cfull2(:,:,t,:) = Cfull(:,:,tmp_sli,:);
end

%%% NOTE!! the "for" and "parfor" loops should have identical code
if is_in_parallel
    for t = 1:tls
        
        tmp_sli = sli_traj(t) + pad;
        tmp_tse_traj = shot_traj{t};
        
        dx = dx_v(t);
        dy = dy_v(t);
        dz = dz_v(t);
        yaw    = yaw_v(t);
        pitch  = pitch_v(t);
        roll   = roll_v(t);
        
        % R
        temp_vol = zeros(nlin,ncol,nsli);
        temp_vol(nz_pxls_in) = x;
        temp_volR = rot3d_v10(temp_vol,yaw,pitch,roll);
        
        nz_pxls_rot = find(repmat(sum(abs(temp_volR),3), 1, 1, nsli));
        nz_pxls_shot = union(nz_pxls_in,nz_pxls_rot);
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_shot) = 1;
        nz_im = sum(nz_im,3);
        sli_pxls_shot = find(nz_im);
        
        Rx = reshape(temp_volR(nz_pxls_shot),numel(sli_pxls_shot),nsli);
        
        % Fz
        FzRx = fftshift(fft(ifftshift(Rx,2), nsli, 2) ,2);
        
        % Mz
        p_ph = exp(-1i * kp_vec * dz).';
        MzFzRx = permute(repmat(p_ph,1,numel(sli_pxls_shot)),[2 1]) .* FzRx;
        
        % Fzin
        FzinMzFzRx = fftshift(ifft(ifftshift(MzFzRx, 2), nsli, 2) ,2);
        
        % Uss
        UssFzinMzFzRx = zeros(nlin*ncol,1);
        UssFzinMzFzRx(sli_pxls_shot,:) = FzinMzFzRx(:,tmp_sli);
        UssFzinMzFzRx = reshape(UssFzinMzFzRx,nlin,ncol);
        
        % Fxy
        FxyUssFzinMzFzRx = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(UssFzinMzFzRx,1),2)),1),2);
        
        % Mxy
        % create motion matrix (assume standard cartesian sampling)
        mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
        Mxy_sli = exp(-1i * sum(kspace_2d.*mmtx_sli,3) );
        MxyFxyUssFzinMzFzRx= Mxy_sli .* FxyUssFzinMzFzRx;
        
        % Fxyin
        FxyinMxyFxyUssFzinMzFzRx = fftshift(fftshift(ifft2(...
            ifftshift(ifftshift(MxyFxyUssFzinMzFzRx,1),2)),1),2);
        
        % C
        Cx = squeeze(Cfull2(:,:,t,:)) .* repmat(FxyinMxyFxyUssFzinMzFzRx,1,1,ncha);
        
        % Fen
        FenCx = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(Cx,1),2) ) ,1),2);
        
        % Uu*Uu
        FMx_input = FenCx;
        FMx_input(setdiff(1:nlin,tmp_tse_traj),:,:) = 0;
        
        %%% Fen* operator
        FensUusk = fftshift(fftshift( ifft2(ifftshift(ifftshift(FMx_input,1),2)) ,1),2);
        
        %%% C* operator
        CsFensUusk = sum(conj(squeeze(Cfull2(:,:,t,:))) .* FensUusk,3);
        
        %%% Fxyin* operator (part of Txy*)
        FxyinsCsFensUssk=fftshift(fftshift(fft2(ifftshift(ifftshift(CsFensUusk,1),2)),1),2);
        
        %%% Mxy* operator (part of Txy*)
        mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
        Mxy_sli = exp(1i * sum(kspace_2d.*mmtx_sli,3) );
        MxysFxyinsCsFensUssk = Mxy_sli .* FxyinsCsFensUssk;
        
        %%% Fxy* operator (part of Txy*)
        FxysMxysFxyinsCsFensUssk = fftshift(fftshift(  ifft2(...
            ifftshift(ifftshift(  MxysFxyinsCsFensUssk, 1), 2)), 1), 2);
        
        %%% Uss* operator
        tmp_imsp = zeros(nlin,ncol,nsli);
        tmp_imsp(:,:,tmp_sli) = FxysMxysFxyinsCsFensUssk;
        
        % prep before Tz
        temp_vol2 = zeros(nlin,ncol,nsli);
        temp_vol2(nz_pxls_in) = 1;
        temp_vol2 = rot3d_v10(temp_vol2,yaw,pitch,roll);
        nz_pxls_pre_rot = find(temp_vol2);
        nz_pxls_all = union(nz_pxls_pre_rot,nz_pxls_in); % not sure if we need nz_pxls_in here
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_all) = 1;
        sli_pxls_all = find(sum(nz_im,3));
        
        %%% Fz-* operator (part of Tz*)
        tmp_imsp2 = reshape(tmp_imsp,nlin*ncol,nsli);
        tmp_imsp2 = tmp_imsp2(sli_pxls_all,:);
        obj_xy_kz = fftshift(fft(ifftshift(tmp_imsp2,2), nsli, 2),2);
        
        %%% Mz* operator (part of Tz*)
        p_ph = exp(1i * kp_vec     * dz).';        % par phase
        Mz_obj_xy_kz = permute(repmat(p_ph,1,numel(sli_pxls_all)),[ 2 1]) .* obj_xy_kz;
        
        %%% Fz* operator (part of Tz*)
        obj_xyz = fftshift(ifft(ifftshift(Mz_obj_xy_kz,2), nsli, 2) ,2);
        
        %%% R* operator
        temp_vol = zeros(nlin*ncol,nsli);
        temp_vol(sli_pxls_all, :) = obj_xyz;
        temp_vol = reshape(temp_vol, nlin, ncol, nsli);
        
        inv_rot_vol = rot3d_v10(temp_vol,-1*yaw,-1*pitch,-1*roll);
        
        xprj_all(:,t) = reshape(inv_rot_vol(nz_pxls_in),numel(nz_pxls_in),1);
    end
else
    parfor t = 1:tls
        
        tmp_sli = sli_traj(t) + pad;
        tmp_tse_traj = shot_traj{t};
        
        dx = dx_v(t);
        dy = dy_v(t);
        dz = dz_v(t);
        yaw    = yaw_v(t);
        pitch  = pitch_v(t);
        roll   = roll_v(t);
        
        % R
        temp_vol = zeros(nlin,ncol,nsli);
        temp_vol(nz_pxls_in) = x;
        temp_volR = rot3d_v10(temp_vol,yaw,pitch,roll);
        
        nz_pxls_rot = find(repmat(sum(abs(temp_volR),3), 1, 1, nsli));
        nz_pxls_shot = union(nz_pxls_in,nz_pxls_rot);
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_shot) = 1;
        nz_im = sum(nz_im,3);
        sli_pxls_shot = find(nz_im);
        
        Rx = reshape(temp_volR(nz_pxls_shot),numel(sli_pxls_shot),nsli);
        
        % Fz
        FzRx = fftshift(fft(ifftshift(Rx,2), nsli, 2) ,2);
        
        % Mz
        p_ph = exp(-1i * kp_vec * dz).';
        MzFzRx = permute(repmat(p_ph,1,numel(sli_pxls_shot)),[2 1]) .* FzRx;
        
        % Fzin
        FzinMzFzRx = fftshift(ifft(ifftshift(MzFzRx, 2), nsli, 2) ,2);
        
        % Uss
        UssFzinMzFzRx = zeros(nlin*ncol,1);
        UssFzinMzFzRx(sli_pxls_shot,:) = FzinMzFzRx(:,tmp_sli);
        UssFzinMzFzRx = reshape(UssFzinMzFzRx,nlin,ncol);
        
        % Fxy
        FxyUssFzinMzFzRx = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(UssFzinMzFzRx,1),2)),1),2);
        
        % Mxy
        % create motion matrix (assume standard cartesian sampling)
        mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
        Mxy_sli = exp(-1i * sum(kspace_2d.*mmtx_sli,3) );
        MxyFxyUssFzinMzFzRx= Mxy_sli .* FxyUssFzinMzFzRx;
        
        % Fxyin
        FxyinMxyFxyUssFzinMzFzRx = fftshift(fftshift(ifft2(...
            ifftshift(ifftshift(MxyFxyUssFzinMzFzRx,1),2)),1),2);
        
        % C
        Cx = squeeze(Cfull2(:,:,t,:)) .* repmat(FxyinMxyFxyUssFzinMzFzRx,1,1,ncha);
        
        % Fen
        FenCx = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(Cx,1),2) ) ,1),2);
        
        % Uu*Uu
        FMx_input = FenCx;
        FMx_input(setdiff(1:nlin,tmp_tse_traj),:,:) = 0;
        
        %%% Fen* operator
        FensUusk = fftshift(fftshift( ifft2(ifftshift(ifftshift(FMx_input,1),2)) ,1),2);
        
        %%% C* operator
        CsFensUusk = sum(conj(squeeze(Cfull2(:,:,t,:))) .* FensUusk,3);
        
        %%% Fxyin* operator (part of Txy*)
        FxyinsCsFensUssk=fftshift(fftshift(fft2(ifftshift(ifftshift(CsFensUusk,1),2)),1),2);
        
        %%% Mxy* operator (part of Txy*)
        mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
        Mxy_sli = exp(1i * sum(kspace_2d.*mmtx_sli,3) );
        MxysFxyinsCsFensUssk = Mxy_sli .* FxyinsCsFensUssk;
        
        %%% Fxy* operator (part of Txy*)
        FxysMxysFxyinsCsFensUssk = fftshift(fftshift(  ifft2(...
            ifftshift(ifftshift(  MxysFxyinsCsFensUssk, 1), 2)), 1), 2);
        
        %%% Uss* operator
        tmp_imsp = zeros(nlin,ncol,nsli);
        tmp_imsp(:,:,tmp_sli) = FxysMxysFxyinsCsFensUssk;
        
        % prep before Tz
        temp_vol2 = zeros(nlin,ncol,nsli);
        temp_vol2(nz_pxls_in) = 1;
        temp_vol2 = rot3d_v10(temp_vol2,yaw,pitch,roll);
        nz_pxls_pre_rot = find(temp_vol2);
        nz_pxls_all = union(nz_pxls_pre_rot,nz_pxls_in); % not sure if we need nz_pxls_in here
        nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_all) = 1;
        sli_pxls_all = find(sum(nz_im,3));
        
        %%% Fz-* operator (part of Tz*)
        tmp_imsp2 = reshape(tmp_imsp,nlin*ncol,nsli);
        tmp_imsp2 = tmp_imsp2(sli_pxls_all,:);
        obj_xy_kz = fftshift(fft(ifftshift(tmp_imsp2,2), nsli, 2),2);
        
        %%% Mz* operator (part of Tz*)
        p_ph = exp(1i * kp_vec     * dz).';        % par phase
        Mz_obj_xy_kz = permute(repmat(p_ph,1,numel(sli_pxls_all)),[ 2 1]) .* obj_xy_kz;
        
        %%% Fz* operator (part of Tz*)
        obj_xyz = fftshift(ifft(ifftshift(Mz_obj_xy_kz,2), nsli, 2) ,2);
        
        %%% R* operator
        temp_vol = zeros(nlin*ncol,nsli);
        temp_vol(sli_pxls_all, :) = obj_xyz;
        temp_vol = reshape(temp_vol, nlin, ncol, nsli);
        
        inv_rot_vol = rot3d_v10(temp_vol,-1*yaw,-1*pitch,-1*roll);
        
        xprj_all(:,t) = reshape(inv_rot_vol(nz_pxls_in),numel(nz_pxls_in),1);
    end
end

xprj = sum(xprj_all,2);

end
