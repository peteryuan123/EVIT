data = load('/home/mpl/data/EVIT/offline/events.txt');


for i=1:20000
    scatter3(data(i, 0), data(i, 1), data(i, 2));
end