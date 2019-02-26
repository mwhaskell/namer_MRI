function [kdata] = A_v10(x,U,Cfull,tse_traj_in,Ms,nz_pxls_in,pad)

%% create parallel pool if needed
if ~is_in_parallel
    % init pool
    np = feature('numCores');
    p = gcp('nocreate'); % If pool, do not create new one.
    if isempty(p)
        parpool(np);
    end
end

%% determine if there are multiple timepoints with the same motion and
% slice, and remove multiples
Ms_plus_mot = cat(2,tse_traj_in(:,1),Ms);
[~,unique_indx,all_indx]=unique(Ms_plus_mot,'rows','stable');
Ms = Ms(unique_indx,:);

tse_traj_all = tse_traj_in;
tse_traj_mtx = tse_traj_in(:,2:end);
tse_traj_mtx_all = tse_traj_mtx;

tse_traj = tse_traj_in(unique_indx,:);

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

% find sequence parameters
tls = size(tse_traj,1);

%% begin forward model

% FMx for "forward model x"
FMx = zeros(size(U));
FenCx_mtx = zeros(nlin,ncol,ncha,tls);


kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);

dx_v = Ms(:,1);dy_v = Ms(:,2);dz_v = Ms(:,3);
yaw_v = Ms(:,4); pitch_v = Ms(:,5); roll_v = Ms(:,6);


Cfull2 = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    tmp_sli = tse_traj(t,1) + pad;
    Cfull2(:,:,t,:) = Cfull(:,:,tmp_sli,:);
end

%% loop over shots, either in parallel or serially depening on how A was called
%   NOTE!! the contents of both these loops should be the same, the only
%   difference should be that one is a "for" and one is a "parfor" loop
if is_in_parallel
    for t = 1:tls
        tmp_sli = tse_traj(t,1) + pad;
        
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
        FenCx_mtx(:,:,:,t) = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(Cx,1),2) ) ,1),2);
        
    end
else  %%% parallel version
    parfor t = 1:tls
        tmp_sli = tse_traj(t,1) + pad;
        
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
        FenCx_mtx(:,:,:,t) = fftshift(fftshift(fft2(...
            ifftshift(ifftshift(Cx,1),2) ) ,1),2);
        
    end
    
end  % end shot loop



%%% update 18-12-18
for t = 1:size(tse_traj_all,1)
    tmp_sli = tse_traj_all(t,1) + pad;
    tmp_tse_traj = tse_traj_mtx_all(t,:);
    unique_indx = all_indx(t);
    
    % Uss
    FMx(tmp_tse_traj,:,tmp_sli,:) = FenCx_mtx(tmp_tse_traj,:,:,unique_indx);
end

kdata = FMx;

end
