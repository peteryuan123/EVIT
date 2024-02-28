info = load("/home/mpl/data/EVIT/result/robot_fast_result/neutral/info.txt");
img = imread("/home/mpl/data/EVIT/result/robot_fast_result/neutral/1642661814.716115.jpg");
img_3channel = cat(3, img, img, img);

rows = size(img, 1);
cols = size(img, 2);

img_f = figure('units','normalized','outerposition',[0 0 1 1]);
plot_f = figure;

max_step = 15;
step_interval = 1;

for i=1:length(info)
    r = info(i, 1);
    c = info(i, 2);
    dx = info(i, 3);
    dy = info(i, 4);
    
    if (int32(r + max_step * dy) <= 0 || int32(r + max_step * dy) > rows || int32(c + max_step * dx) <= 0 || int32(c + max_step * dx) > cols)
        continue
    end

    temp = img_3channel;

    data = [];
    for step=max_step:-step_interval:0
        r_sample = int32(r + dy * step);
        c_sample = int32(c + dx * step);
        temp(r_sample, c_sample, :) = [0, 0, 255];
        value_sample = img(r_sample, c_sample);
        data = [data, value_sample];
    end
    
    figure(plot_f);
    plot(data);
    
    figure(img_f);
    imshow(temp);
    waitforbuttonpress;
    img_3channel(info(i, 1), info(i, 2), :) = [255, 0, 0];
end




imshow(img_3channel)
% 
% for i=1:length(data)
%     plot(data(i, :));
%     waitforbuttonpress;
% end