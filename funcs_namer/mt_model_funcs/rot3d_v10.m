function [rvol] = rot3d_v10(vol,yaw,pitch,roll)

temp_vol = vol;
if roll
    temp_vol = imrotate(temp_vol,roll,'bilinear','crop');
end
if yaw
    temp_vol = permute( ...
        imrotate(permute(temp_vol, [3 2 1]),yaw,'bilinear','crop'), [3 2 1]);
end
if pitch
    temp_vol = permute( ...
        imrotate(permute(temp_vol, [3 1 2]),pitch,'bilinear','crop'), [2 3 1]);
end

rvol = temp_vol;

end
