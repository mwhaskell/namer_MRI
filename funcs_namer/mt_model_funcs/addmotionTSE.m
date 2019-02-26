function [ output_kdata ] = MHaddmotionTSE_gen( kU_hy, mmtx, tse_traj, pad )
% this function takes an image "input_image" and a matrix of motion (of
% size Nx6) and outputs a motion corrupted image "output_image"

% input can be padded with zero slices before and after to avoid leaking
% into other slices, and is assumed to be the same amount of zeros slices
% before the data begins

% "gen" because it can take any tse trajectory, doesn't assume a pattern
% based on the first line sampled and the TF

% tse_traj is a tls x TF + 1 (total shots by turbo factor + 1) matrix
% determinging the slice order (tse_traj(:,1)) and the line trajectory
% (tse_traj(:,2:end)) at each time point

% mmtx is a tls x 3 matrix with the rows representing the motion at each
% shot, (dP,dL,dI,rA,rL,rI)

%%% last updated: 3/30/16

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                     %
%   The motion params must be in the oder:            %
%       1. dP   translation posterior                 %
%       2. dL   translation left                      %
%       3. dI   translation inferior                  %
%       4. rA   RH rotation about anterior axis       %
%       5. rL   RH rotation about left axis           %
%       6. rI   RH rotation about inferior axis       %
%                                                     %
%                                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% note! the inferior shift is in unit of pixels (slice thickness) for the
%%% third dimension,
%%% so it may have different values from dP and dL for the same physical
%%% motion if the resolution is not isotropic in all directions

% find image and TSE parameters
[nlin, ncol, nsli] = size(kU_hy);
TF = size(tse_traj,2) - 1;
tls = size(tse_traj,1);
% sps = nlin/(R*TF);
% pad = ( nsli - tls/sps )/2;

% check that mmtx, tse_traj and input_image have corresponding sizes
if tls ~= size(mmtx,1)
    disp(tls)
    disp(size(mmtx,1))
    err % motion matrix incorrect length
end
if max(abs(mmtx(:))) == 0,
    output_kdata = kU_hy;
    return
end

output_kdata = zeros(size(kU_hy));

kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol);


% find object in image space and x-y-kz hybrid space
obj_xyz = ifft2c(kU_hy);

% loop through shots and add motion
for ii = 1:tls
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%   get parameters, check for          no motion  %%%%%%%%%
    
    tmp_sli = tse_traj(ii,1) + pad;
    tmp_tse_traj = tse_traj(ii,2:end);
    
    delr = mmtx(ii,1);      % shift posterior in pxl
    delc = mmtx(ii,2);      % shift left in pxl
    delp = mmtx(ii,3);      % shift inferior in pxl
    
    yaw    = mmtx(ii,4);
    pitch  = mmtx(ii,5);
    roll   = mmtx(ii,6);
    rotation_cutoff = .2;
    if abs(yaw) < rotation_cutoff, yaw = 0; end
    if abs(pitch) < rotation_cutoff, pitch = 0; end
    if abs(roll) < rotation_cutoff, roll = 0; end
    
    if ~delr && ~delc && ~delp && ~yaw && ~pitch && ~roll
        output_kdata(tmp_tse_traj,:,tmp_sli) = kU_hy(tmp_tse_traj,:,tmp_sli);
        continue
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%     rotation motion                             %%%%%%%%%
    
    temp_vol = obj_xyz;
    % rotate all frames along each axis in image space
    if roll
        for pp = 1:nsli
            temp_vol(:,:,pp) = imrotate(temp_vol(:,:,pp),roll,'bilinear','crop');
        end
    end
    if yaw
        for ll = 1:nlin
            temp_cor_view =  imrotate(transpose(squeeze(temp_vol(ll,:,:))),yaw,'bilinear','crop');
            temp_vol(ll,:,:) = transpose(temp_cor_view);
        end
    end
    if pitch
        for cc = 1:ncol
            temp_sag_view = imrotate(transpose(squeeze(temp_vol(:,cc,:))),pitch,'bilinear','crop');
            temp_vol(:,cc,:) = transpose(temp_sag_view);
        end
    end
    
    Robj_xyz = temp_vol;
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%     z motion (out of plane motion)              %%%%%%%%%
    
    obj_xy_kz = fftc(Robj_xyz, nsli, 3);
    
    % add phase ramp to hybrid space of object
    kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
    p_ph = exp(-1i * kp_vec     * delp).';        % par phase
    Mz_obj_xy_kz = permute(repmat(p_ph,1,ncol,nlin),[3 2 1]) .* obj_xy_kz;
    
    % inverse fourier transform along the z dimension
    obj_post_Mz = ifftc(Mz_obj_xy_kz, nsli, 3);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%     xy motion (in-plane motion)                 %%%%%%%%%
    
    % perform slice selective 2d fourier transform
    kU_hy2 = fft2c(obj_post_Mz);
    
    % calculate phase ramps for in plane translations
    r_ph = exp(-1i * kr_vec     * delr).';      % row phase
    c_ph = exp(-1i * kc_vec     * delc);        % col phase (readout)
    
    % select proper TSE readout lines
    rc_ph = r_ph(tmp_tse_traj) * c_ph;
    output_kdata(tmp_tse_traj,:,tmp_sli) = (rc_ph .* squeeze(kU_hy2(tmp_tse_traj,:,tmp_sli)));
    
end




end