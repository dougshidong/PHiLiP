clear;
syms x y z a b c d e f

%u1d = @(x,y,z,a,b,c,d,e,f) (sin(a*x+d));
%u2d = @(x,y,z,a,b,c,d,e,f) (sin(a*x+d)*sin(b*y+e));
%u3d = @(x,y,z,a,b,c,d,e,f) (sin(a*x+d)*sin(b*y+e)*sin(c*z+f));


u1d = @(x,y,z,a,b,c,d,e,f) (cos(a*x+d));
u2d = @(x,y,z,a,b,c,d,e,f) (cos(a*x+d)*cos(b*y+e));
u3d = @(x,y,z,a,b,c,d,e,f) (cos(a*x+d)*cos(b*y+e)*sin(c*z+f));

%% Integrate from 0 to 1 the manufactured solution int_0^1 u dx dy dz
f = @(x,y,z,a,b,c,d,e,f) u1d(x,y,z,a,b,c,d,e,f);
int(f,x); % -cos(d + a*x)/a
int(f,x, 0, 1) % -(cos(a + d) - cos(d))/a

f = @(x,y,z,a,b,c,d,e,f) u2d(x,y,z,a,b,c,d,e,f);
int(f,x,y); % (sin(e + b*y)*(cos(d + a*x) - cos(d + a*y)))/a
int( int(f,x,0,1) ,y,0,1) % ((cos(a + d) - cos(d))*(cos(b + e) - cos(e)))/(a*b)

f = @(x,y,z,a,b,c,d,e,f) u3d(x,y,z,a,b,c,d,e,f);
int(f,x,y,z); % (sin(e + b*y)*sin(f + c*z)*(cos(d + a*y) - cos(d + a*z)))/a
int( int( int(f,x,0,1) ,y,0,1), z,0,1) % -((cos(a + d) - cos(d))*(cos(b + e) - cos(e))*(cos(c + f) - cos(f)))/(a*b*c)

%% Integrate from 0 to 1 the manufactured solution int_0^1 u^2 dx dy dz
f = @(x,y,z,a,b,c,d,e,f) u1d(x,y,z,a,b,c,d,e,f)^2;
% x/2 - sin(2*d + 2*a*x)/(4*a)
int(f,x);
% (sin(2*d)/4 - sin(2*a + 2*d)/4)/a + 1/2
int(f,x, 0, 1)

f = @(x,y,z,a,b,c,d,e,f) u2d(x,y,z,a,b,c,d,e,f)^2;
% sin(e + b*y)^2*(y/2 - x/2 + (sin(2*d + 2*a*x)/4 - sin(2*d + 2*a*y)/4)/a)
int(f,x,y);
% ((2*a + sin(2*d) - sin(2*a + 2*d))*(2*b + sin(2*e) - sin(2*b + 2*e)))/(16*a*b)
int( int(f,x,0,1) ,y,0,1)

f = @(x,y,z,a,b,c,d,e,f) u3d(x,y,z,a,b,c,d,e,f)^2;
% sin(e + b*y)^2*sin(f + c*z)^2*(z/2 - y/2 + (sin(2*d + 2*a*y)/4 - sin(2*d + 2*a*z)/4)/a)
int(f,x,y,z);
%((2*a + sin(2*d) - sin(2*a + 2*d))*(2*b + sin(2*e) - sin(2*b + 2*e))*(2*c + sin(2*f) - sin(2*c + 2*f)))/(64*a*b*c)
int( int( int(f,x,0,1) ,y,0,1), z,0,1)

