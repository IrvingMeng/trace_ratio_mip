% plot the selected features in the image
features_mio = [   9,   11,   14,   18,   22,   26,   30,   35,   38,   49,   51,...
         53,   55,   56,   72,   74,   76,   80,   96,  137,  142,  146,...
        149,  153,  157,  163,  165,  167,  181,  183,  200,  203,  205,...
        207,  210,  213,  216,  219,  223,  225,  228,  230,  249,  265,...
        266,  268,  272,  277,  280,  282,  284,  286,  288,  290,  292,...
        344,  346,  348,  350,  352,  354,  356,  412,  414,  416,  418,...
        419,  481,  522,  523,  524,  544,  545,  546,  585,  586,  587,...
        588,  589,  844,  845,  846,  848,  849,  907,  908,  909, 1293,...
       1612, 1614, 1674, 1675, 1677, 1740, 2060, 2187, 2250, 2377, 2440,...
       2505]+1;
features_naive = [148, 150, 149, 151,  19,  83,  84,  85, 147,  87,  86,  20,  18,...
        88, 152,  23,  82, 215, 153, 217, 214,  21,  24, 216, 146,  89,...
        25,  22, 145,  26,  17, 218,  81, 154, 281, 213,  27,  90,  80,...
        91,  92, 144,  16,  28,  14,  13,  15, 212,  79, 282, 143, 155,...
       142,  78, 219,  93,  77, 211,  12,  29, 157, 141, 156,  76, 280,...
       283,  30,  94,  11, 210, 208, 158, 220,  75,  31, 140,  96,  32,...
       209,  33, 221, 206, 222,  95, 139, 279, 207, 908,  34, 284,  97,...
       160, 346, 161, 159, 278, 286, 223, 225,  10]+1;
features_nie = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,... 
    27, 28, 29, 30, 31, 32, 33, 34, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, ...
    86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 139, 140, 141, 142, 143, 144, ...
    145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, ...
    161, 163, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, ...
    220, 221, 222, 223, 225, 279, 280, 281, 282, 283, 284, 286, 287, 289, 346]+1;



num = 100;
I = imresize(imread('./data/orl_face/1.pgm'), [32, 32]);
imshow(I);
set(gca,'DataAspectRatioMode','auto')
set(gca,'Position',[0 0 1 1])
pos = get(gcf,'pos');
set(gcf,'pos',[pos(1) pos(2) 256 256])
hold on;
for i = 1:num
    x = fix(features_naive(i)/32)+1;
    y = mod(features_naive(i),32);
    plot(x,y,'rO','MarkerSize',3);
end

imshow(I);
set(gca,'DataAspectRatioMode','auto')
set(gca,'Position',[0 0 1 1])
pos = get(gcf,'pos');
set(gcf,'pos',[pos(1) pos(2) 256 256])
hold on;
for i = 1:num
    x = fix(features_nie(i)/32)+1;
    y = mod(features_nie(i),32);
    plot(x,y,'rO','MarkerSize',3);
end

imshow(I);
set(gca,'DataAspectRatioMode','auto')
set(gca,'Position',[0 0 1 1])
pos = get(gcf,'pos');
set(gcf,'pos',[pos(1) pos(2) 256 256])
hold on;
for i = 1:num
    x = fix(features_mio(i)/32)+1;
    y = mod(features_mio(i),32);
    plot(x,y,'rO','MarkerSize',3);
end
