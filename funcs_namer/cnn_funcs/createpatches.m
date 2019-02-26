function [patch_ims] = createpatches(full_im, patch_sz, patch_stsz)


[nrow, ncol, nims] = size(full_im);
patch_hs = (patch_sz - 1) / 2;  % patch half size
rindx = patch_hs:patch_stsz:nrow-patch_hs-1;
cindx = patch_hs:patch_stsz:ncol-patch_hs-1;
npatch = nims * numel(rindx) * numel(cindx);
patch_ims = zeros(patch_sz, patch_sz, npatch);

indx = 1;
for nn = 1:nims
    for x = 1:numel(rindx)
        for y = 1:numel(cindx)
            patch = full_im(rindx(x)+1-patch_hs: rindx(x) + patch_hs + 1,...
                cindx(y)+1-patch_hs: cindx(y) + patch_hs + 1, nn);
            patch_ims(:, :, indx) = patch;
            indx = indx + 1;
        end
    end
end


end
