function [  ] = mosaic( imgs, row_num, col_num, fig_num, title_str, disp_range )

% cat 2d images together as given in row_num, col_num


% convert to abs if input is complex

imag_part = imag(imgs(:));
if norm(imag_part) ~=0 
    imgs = abs(imgs);
end
    

if row_num * col_num ~= size(imgs,3)
    %disp('In tile_pdf: sizes do not match, auto correction')
    
    if row_num * col_num > size(imgs,3)
        % zero pad the image
        img_add = zeros(size(imgs(:,:,1)));
        imgs = cat(3, imgs, repmat(img_add,[1,1,row_num * col_num - size(imgs,3)]) ); 
    end
    
    
end

show_ = zeros([size(imgs,1)*row_num, size(imgs,2)*col_num]);


for r = 1:row_num
    S = imgs(:,:,col_num*(r-1)+1);
    
    for c = 2:col_num
        S = cat(2, S, imgs(:,:,col_num*(r-1)+c));
    end
    
    if r == 1
        show_ = S;
    else
        show_ = cat(1, show_, S);
    end
end


if nargin < 6
    disp_range(1) = min(show_(:));
    disp_range(2) = max(show_(:));
end

if nargin < 4
    figure, imagesc(show_, disp_range), axis image off, colormap gray
else
    figure(fig_num), imagesc(show_, disp_range), axis image off, colormap gray
end


if nargin >= 5
    title(title_str)
end

drawnow


end

