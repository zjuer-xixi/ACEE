%代码如下
N=-0.01:0.0001:0.01;
[x,y]=meshgrid(N);
[theta,rho]=cart2pol(x,y);
%光斑尺寸常数
w0=1e-3;                              
%随机确定l、p值
l=3;p=0;                               
%为简化操作，选取z=0 
z=0;                                   
%光的波长、波数
t=0.632e-6;k=2*pi/t;                      
%瑞利距离
RZ=pi*w0^2/t; 
%光束在r处半径
w_z = w0*sqrt(1+(z/RZ)^2);                         
%计算公式
u=sqrt(2*factorial(p)/pi/(p+factorial(abs(l))))/w_z*(sqrt(2)*rho/w_z).^abs(l)...
.*exp(-rho.^2/w_z^2).*Laguerre(p,abs(l),2*rho.^2/w_z^2).*exp(-1i*l*theta).*exp(-1i*k*z)...
.*exp(-1i*k*rho.^2/2/RZ)*exp(-1i*(2*p+abs(l)+1)*atan(z/RZ));
%计算光强
I2=u.*conj(u);                          
Ie=sqrt(I2);                           
%画光强相位
figure
mesh(x,y,Ie);
shading interp                
zlabel('intensity');

figure
vortex_phase=rem(angle(u)+2*pi,2*pi);
imshow(vortex_phase,[])
