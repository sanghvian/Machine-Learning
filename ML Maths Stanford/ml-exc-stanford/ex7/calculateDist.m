function dist = calculateDist(p1,p2)
%CALCULATEDIST Summary of this function goes here
%   Detailed explanation goes here
        dim = size(p2,2);
        dist =  sum((p1 - p2).^2);           
end

