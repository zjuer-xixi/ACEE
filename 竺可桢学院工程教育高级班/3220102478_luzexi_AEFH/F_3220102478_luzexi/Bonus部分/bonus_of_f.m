% 导入数据点
x = load("data.txt");

% 进行k-means聚类
k = 10;
[idx, C] = kmeans(x, k);

% 绘制散点图并对不同类别的点进行着色
figure;
scatter3(x(:,1), x(:,2), x(:,3), 30, idx);


