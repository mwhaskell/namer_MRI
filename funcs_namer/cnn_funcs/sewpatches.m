function [ output_im] = sewpatches(input_patches, output_sz, patch_sz, patch_stsz)


nrow = output_sz(1);
ncol = output_sz(2);
patch_hs = (patch_sz - 1) / 2;  % patch half size
rindx = patch_hs:patch_stsz:nrow-patch_hs;
cindx = patch_hs:patch_stsz:ncol-patch_hs;

psf_img = zeros(nrow, ncol);
res_img = zeros(nrow, ncol);

indx = 1;
for x = 1:numel(rindx)
    for y = 1:numel(cindx)
        patch = reshape(input_patches(indx,:,:),[patch_sz, patch_sz]);
        
        psf_img(rindx(x)+1-patch_hs:rindx(x)+patch_hs+1,...
            cindx(y)+1-patch_hs:cindx(y)+patch_hs+1) = ...
            psf_img(rindx(x)+1-patch_hs:rindx(x)+patch_hs+1,...
            cindx(y)+1-patch_hs:cindx(y)+patch_hs+1) + 1;
        res_img(rindx(x)+1-patch_hs:rindx(x)+patch_hs+1,...
            cindx(y)+1-patch_hs:cindx(y)+patch_hs+1) = ...
            res_img(rindx(x)+1-patch_hs:rindx(x)+patch_hs+1,...
            cindx(y)+1-patch_hs:cindx(y)+patch_hs+1) + patch;
        
        indx = indx + 1;
    end
end

output_im = res_img ./ (psf_img + 1e-12);

end
