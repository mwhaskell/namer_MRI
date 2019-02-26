function [tse_mt_cell_compact] = tse_traj_to_compact_cell(tse_traj,Ms)
% this function converts a tse_traj to a compact version where all shots at
% a given slice and motion state are combined

% This will combine all of
% the lines from a given slice that had the same motion into one shot (i.e.
% if tse_traj was originally 17x11 but the first two shots had the same
% motion parameters (and the rest were unique) the row element of the cell
% would contain a 22x1 vector with the lines from shots 1 and 2. The, the
% next 15 elements of the cell would contain 11x1 vectors with the lines
% from the rest of the shots.

tls = size(tse_traj,1);
tse_mt_cell0 = cell(tls,3);
for ii = 1:tls

    tse_mt_cell0{ii,1} = tse_traj(ii,1);
    tse_mt_cell0{ii,2} = tse_traj(ii,2:end);
    tse_mt_cell0{ii,3} = Ms(ii,:);

end

tse_mt_cell_compact = tse_mt_cell0;
ind = 1;
unique_sli_plus_mt = [];
while ind <= size(tse_mt_cell_compact,1)

    csli = tse_mt_cell_compact{ind,1};
    clin = tse_mt_cell_compact{ind,2};
    cmt = tse_mt_cell_compact{ind,3};
    
    csli_plus_mt = [csli, cmt];
    if ind == 1
        unique_sli_plus_mt = csli_plus_mt;
        ind = ind + 1;
        continue
    end
    if ~nnz(ismember(csli_plus_mt,unique_sli_plus_mt,'rows'))
        % add unique combination of slice and motion state to list, and
        % don't make any changes to tse_mt_cell_compact because this is a
        % unique motion state for that slice
        unique_sli_plus_mt = [unique_sli_plus_mt;csli_plus_mt];
        ind = ind + 1;
    else
        [~,row] = ismember(csli_plus_mt,unique_sli_plus_mt,'rows');
        tse_mt_cell_compact{row,2} = [tse_mt_cell_compact{row,2},clin];
        
        tse_mt_cell_compact(ind,:) = [];
        
    end
    
end


end