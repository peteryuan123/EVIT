% img = imread("/home/mpl/data/EVIT/result/robot_fast_result/exp/1642661814.506115.jpg");
b = bar3(img2);
colorbar
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end